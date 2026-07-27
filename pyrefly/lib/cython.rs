/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use lsp_types::CompletionItem;
use lsp_types::CompletionItemKind;
use ruff_text_size::TextRange;
use ruff_text_size::TextSize;
use tree_sitter::Node;
use tree_sitter::Parser;

use crate::module::module_info::ModuleInfo;

const CYTHON_EXTENSIONS: &[&str] = &["pyx", "pxd", "pxi"];

pub(crate) struct LoweredCython {
    pub(crate) source: String,
    pub(crate) prelude: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CompileTimeValue {
    Bool(bool),
    Int(i64),
    String(String),
}

#[derive(Default)]
struct LoweringContext {
    compile_time_values: HashMap<String, CompileTimeValue>,
    fused_types: Vec<(String, Vec<String>)>,
}

pub(crate) fn is_cython_module(module: &ModuleInfo) -> bool {
    module
        .path()
        .as_path()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| CYTHON_EXTENSIONS.contains(&ext))
}

pub(crate) fn syntax_error_ranges(contents: &str) -> Vec<TextRange> {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_cython::language())
        .is_err()
    {
        return Vec::new();
    }
    let Some(tree) = parser.parse(contents, None) else {
        return Vec::new();
    };
    let mut ranges = BTreeSet::new();
    collect_error_ranges(tree.root_node(), &mut ranges);
    ranges
        .into_iter()
        .map(|(start, end)| TextRange::new(start, end))
        .collect()
}

/// Lower the Cython constructs we understand to Python syntax, while keeping every
/// line the same byte length so diagnostics in the shared type checker still point
/// into the original file.
pub(crate) fn lower_to_python(contents: &str) -> LoweredCython {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cython::language())
        .expect("the bundled Cython grammar must be loadable");
    let Some(tree) = parser.parse(contents, None) else {
        return LoweredCython {
            source: String::new(),
            prelude: String::new(),
        };
    };
    let mut lowered = contents.as_bytes().to_vec();
    let mut context = LoweringContext::default();
    lower_node(tree.root_node(), contents, &mut lowered, &mut context);

    // Short private aliases keep annotations within the original line lengths. The
    // aliases still resolve to public `cython.*` qualified classes in the type system.
    let mut prelude = concat!(
        "from cython import c_bint as _B, c_double as _D, c_float as _F, ",
        "c_int as _I, MemoryView as _M, Pointer as _P\n",
        "_G: bool = True\n",
        "from typing import TypeVar as _V\n",
    )
    .to_owned();
    for (name, constraints) in context.fused_types {
        prelude.push_str(&format!(
            "{name} = _V({name:?}, {})\n",
            constraints.join(", ")
        ));
    }
    LoweredCython {
        source: String::from_utf8(lowered)
            .expect("lowering preserves the UTF-8 source outside ASCII edits"),
        prelude,
    }
}

pub(crate) fn completion_items(module: &ModuleInfo, position: TextSize) -> Vec<CompletionItem> {
    let contents = module.contents();
    let Some(base) = attribute_base_at(contents, position) else {
        return keyword_completion_items();
    };
    let index = build_index(contents);
    let members = index
        .member_map_for_base(&base)
        .or_else(|| index.member_map_for_type(&base));
    let Some(members) = members else {
        return Vec::new();
    };
    members
        .iter()
        .map(|(name, kind)| CompletionItem {
            label: name.clone(),
            kind: Some(*kind),
            ..Default::default()
        })
        .collect()
}

fn collect_error_ranges(node: Node<'_>, ranges: &mut BTreeSet<(TextSize, TextSize)>) {
    if node.is_error() || node.is_missing() {
        let start = TextSize::new(node.start_byte() as u32);
        let end = TextSize::new(node.end_byte() as u32);
        if start < end {
            ranges.insert((start, end));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_error_ranges(child, ranges);
    }
}

fn lower_node(node: Node<'_>, source: &str, lowered: &mut [u8], context: &mut LoweringContext) {
    match node.kind() {
        "cdef_statement" => {
            lower_cdef_statement(node, source, lowered, context);
            return;
        }
        "cvar_decl" => {
            lower_c_declaration(node, node, source, lowered, context);
            return;
        }
        "DEF_statement" => {
            lower_compile_time_definition(node, source, lowered, context);
            return;
        }
        "IF_statement" => {
            lower_compile_time_if(node, source, lowered, context);
            return;
        }
        "with_statement" if is_gil_context(node, source) => {
            lower_gil_context(node, source, lowered, context);
            return;
        }
        "ELIF_clause" => replace_keyword(node, lowered, b"elif"),
        "ELSE_clause" => replace_keyword(node, lowered, b"else"),
        "ctypedef_statement" => {
            if let Some(fused) = first_descendant_of_kinds(node, &["fused"]) {
                collect_fused_type(fused, source, context);
            }
            replace_with_pass(node, source, lowered);
            return;
        }
        "include_statement" | "extern_block" => {
            replace_with_pass(node, source, lowered);
            return;
        }
        "cimport" => {
            replace_exact(node, lowered, b"import ");
            return;
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        lower_node(child, source, lowered, context);
    }
}

fn lower_cdef_statement(
    node: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let Some(declaration) = first_named_child(node) else {
        replace_with_pass(node, source, lowered);
        return;
    };
    match declaration.kind() {
        "cvar_def" => lower_c_declaration(node, declaration, source, lowered, context),
        "cdef_type_declaration" => {
            let Some(type_declaration) =
                first_descendant_of_kinds(declaration, &["class_definition", "struct"])
            else {
                replace_with_pass(node, source, lowered);
                return;
            };
            lower_class_like(node, type_declaration, source, lowered, context);
        }
        _ => replace_with_pass(node, source, lowered),
    }
}

fn lower_class_like(
    outer: Node<'_>,
    class: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let name = class
        .child_by_field_name("name")
        .or_else(|| first_identifier_node(class));
    let Some(name) = name else {
        replace_with_pass(outer, source, lowered);
        return;
    };
    replace_padded(outer.start_byte(), name.start_byte(), "class ", lowered);
    if let Some(body) = class.child_by_field_name("body").or_else(|| {
        first_child_kind(class, "struct_suite").or_else(|| first_child_kind(class, "block"))
    }) {
        lower_node(body, source, lowered, context);
    }
}

fn lower_c_declaration(
    outer: Node<'_>,
    declaration: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let function = first_child_kind(declaration, "c_function_definition");
    let (type_node, name_node) = if declaration.kind() == "cvar_def" {
        let Some(typed_name) = first_child_kind(declaration, "maybe_typed_name") else {
            replace_with_pass(outer, source, lowered);
            return;
        };
        (
            typed_name.child_by_field_name("type"),
            typed_name.child_by_field_name("name"),
        )
    } else {
        (
            first_child_kind(declaration, "c_type"),
            immediate_identifier_nodes(declaration).into_iter().next(),
        )
    };
    let Some(name_node) = name_node else {
        replace_with_pass(outer, source, lowered);
        return;
    };
    let Some(name) = source.get(name_node.byte_range()) else {
        replace_with_pass(outer, source, lowered);
        return;
    };
    let annotation = if declaration.kind() == "cvar_def" {
        first_child_kind(declaration, "maybe_typed_name")
            .and_then(|typed_name| source.get(typed_name.start_byte()..name_node.start_byte()))
            .map(cython_type_annotation)
            .unwrap_or_else(|| "object".to_owned())
    } else {
        type_node
            .and_then(|x| source.get(x.byte_range()))
            .map(cython_type_annotation)
            .unwrap_or_else(|| "object".to_owned())
    };

    if let Some(function) = function {
        lower_c_function(outer, function, name, &annotation, source, lowered, context);
        return;
    }

    let value = {
        let mut cursor = declaration.walk();
        declaration.named_children(&mut cursor).find(|child| {
            child.start_byte() > name_node.end_byte()
                && !matches!(
                    child.kind(),
                    "identifier" | "type_modifier" | "type_index" | "maybe_typed_name" | "c_type"
                )
        })
    };
    let end = line_content_end(outer, source);
    if let Some(value) = value {
        let prefix = format!("{name}:{annotation}=");
        if prefix.len() <= value.start_byte().saturating_sub(outer.start_byte()) {
            replace_padded(outer.start_byte(), value.start_byte(), &prefix, lowered);
            return;
        }
        let Some(value_text) = source.get(value.byte_range()) else {
            replace_with_pass(outer, source, lowered);
            return;
        };
        let replacement = format!("{prefix}{value_text}");
        if replacement.len() <= end.saturating_sub(outer.start_byte()) {
            replace_padded(outer.start_byte(), end, &replacement, lowered);
            return;
        }
    } else {
        let replacement = format!("{name}:{annotation}");
        if replacement.len() <= end.saturating_sub(outer.start_byte()) {
            replace_padded(outer.start_byte(), end, &replacement, lowered);
            return;
        }
    }
    replace_with_pass(outer, source, lowered);
}

fn lower_c_function(
    declaration: Node<'_>,
    function: Node<'_>,
    name: &str,
    return_annotation: &str,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let Some(parameters) = first_child_kind(function, "c_parameters") else {
        replace_with_pass(declaration, source, lowered);
        return;
    };
    let mut annotated_parameters = Vec::new();
    let mut untyped_parameters = Vec::new();
    let mut cursor = parameters.walk();
    for parameter in parameters.named_children(&mut cursor) {
        if parameter.kind() != "maybe_typed_name" {
            continue;
        }
        let Some(name_node) = parameter.child_by_field_name("name") else {
            continue;
        };
        let Some(parameter_name) = source.get(name_node.byte_range()) else {
            continue;
        };
        untyped_parameters.push(parameter_name.to_owned());
        if parameter.child_by_field_name("type").is_some()
            && let Some(parameter_type) = source.get(parameter.start_byte()..name_node.start_byte())
        {
            annotated_parameters.push(format!(
                "{parameter_name}:{}",
                cython_type_annotation(parameter_type)
            ));
        } else {
            annotated_parameters.push(parameter_name.to_owned());
        }
    }

    let end = first_line_content_end(declaration, source);
    let available = end.saturating_sub(declaration.start_byte());
    let annotated = annotated_parameters.join(",");
    let untyped = untyped_parameters.join(",");
    let candidates = [
        format!("def {name}({annotated})->{return_annotation}:"),
        format!("def {name}({annotated}):"),
        format!("def {name}({untyped}):"),
    ];
    if let Some(replacement) = candidates
        .into_iter()
        .find(|candidate| candidate.len() <= available)
    {
        replace_padded(declaration.start_byte(), end, &replacement, lowered);
        if let Some(body) = first_child_kind(function, "block") {
            lower_node(body, source, lowered, context);
        }
    } else {
        replace_with_pass(declaration, source, lowered);
    }
}

fn lower_compile_time_definition(
    node: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let Some(name) = node
        .child_by_field_name("name")
        .and_then(|x| source.get(x.byte_range()))
    else {
        replace_with_pass(node, source, lowered);
        return;
    };
    let mut cursor = node.walk();
    let value_node = node.named_children(&mut cursor).last();
    let Some(value_node) = value_node else {
        replace_with_pass(node, source, lowered);
        return;
    };
    let Some(value) = source.get(value_node.byte_range()) else {
        replace_with_pass(node, source, lowered);
        return;
    };
    if let Some(value) = evaluate_compile_time_expression(value, &context.compile_time_values) {
        context.compile_time_values.insert(name.to_owned(), value);
    }
    let end = line_content_end(node, source);
    let replacement = format!("{name}={value}");
    replace_padded(node.start_byte(), end, &replacement, lowered);
}

fn lower_compile_time_if(
    node: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let Some(condition) = node.child_by_field_name("condition") else {
        lower_compile_time_if_as_runtime(node, source, lowered, context);
        return;
    };
    let Some(consequence) = node.child_by_field_name("consequence") else {
        lower_compile_time_if_as_runtime(node, source, lowered, context);
        return;
    };

    let mut branches = vec![(Some(condition), consequence, node)];
    let mut cursor = node.walk();
    for alternative in node.children_by_field_name("alternative", &mut cursor) {
        match alternative.kind() {
            "ELIF_clause" => {
                let Some(condition) = alternative.child_by_field_name("condition") else {
                    lower_compile_time_if_as_runtime(node, source, lowered, context);
                    return;
                };
                let Some(consequence) = alternative.child_by_field_name("consequence") else {
                    lower_compile_time_if_as_runtime(node, source, lowered, context);
                    return;
                };
                branches.push((Some(condition), consequence, alternative));
            }
            "ELSE_clause" => {
                let Some(body) = alternative.child_by_field_name("body") else {
                    lower_compile_time_if_as_runtime(node, source, lowered, context);
                    return;
                };
                branches.push((None, body, alternative));
            }
            _ => unreachable!("Cython IF alternatives are ELIF or ELSE clauses"),
        }
    }

    let mut selected = None;
    for (index, (condition, _, _)) in branches.iter().enumerate() {
        match condition {
            Some(condition) => {
                let Some(text) = source.get(condition.byte_range()) else {
                    lower_compile_time_if_as_runtime(node, source, lowered, context);
                    return;
                };
                match evaluate_compile_time_expression(text, &context.compile_time_values) {
                    Some(CompileTimeValue::Bool(true)) => {
                        selected = Some(index);
                        break;
                    }
                    Some(CompileTimeValue::Bool(false)) => {}
                    _ => {
                        lower_compile_time_if_as_runtime(node, source, lowered, context);
                        return;
                    }
                }
            }
            None => {
                selected = Some(index);
                break;
            }
        }
    }

    for (index, (_condition, body, clause)) in branches.into_iter().enumerate() {
        match clause.kind() {
            "IF_statement" => replace_keyword(clause, lowered, b"if"),
            "ELIF_clause" => replace_keyword(clause, lowered, b"elif"),
            "ELSE_clause" => replace_keyword(clause, lowered, b"else"),
            _ => unreachable!("compile-time branch has a known clause kind"),
        }
        if selected == Some(index) {
            lower_node(body, source, lowered, context);
        } else {
            replace_with_pass(body, source, lowered);
        }
    }
}

fn lower_compile_time_if_as_runtime(
    node: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    replace_keyword(node, lowered, b"if");
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        lower_node(child, source, lowered, context);
    }
}

fn is_gil_context(node: Node<'_>, source: &str) -> bool {
    matches!(
        source
            .get(node.start_byte()..first_line_content_end(node, source))
            .map(str::trim),
        Some("with gil:") | Some("with nogil:")
    )
}

fn lower_gil_context(
    node: Node<'_>,
    source: &str,
    lowered: &mut [u8],
    context: &mut LoweringContext,
) {
    let end = first_line_content_end(node, source);
    replace_padded(node.start_byte(), end, "if _G:", lowered);
    let body = node
        .child_by_field_name("body")
        .expect("a Cython gil context always has a body");
    lower_node(body, source, lowered, context);
}

fn evaluate_compile_time_expression(
    expression: &str,
    values: &HashMap<String, CompileTimeValue>,
) -> Option<CompileTimeValue> {
    let expression = expression.trim();
    let expression = expression
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .unwrap_or(expression)
        .trim();
    if let Some(value) = values.get(expression) {
        return Some(value.clone());
    }
    match expression {
        "True" => return Some(CompileTimeValue::Bool(true)),
        "False" => return Some(CompileTimeValue::Bool(false)),
        _ => {}
    }
    if let Some(inner) = expression.strip_prefix("not ")
        && let CompileTimeValue::Bool(value) = evaluate_compile_time_expression(inner, values)?
    {
        return Some(CompileTimeValue::Bool(!value));
    }
    for (operator, equality) in [("==", true), ("!=", false)] {
        if let Some((left, right)) = expression.split_once(operator) {
            let left = evaluate_compile_time_expression(left, values)?;
            let right = evaluate_compile_time_expression(right, values)?;
            return Some(CompileTimeValue::Bool((left == right) == equality));
        }
    }
    if let Ok(value) = expression.parse() {
        return Some(CompileTimeValue::Int(value));
    }
    if expression.len() >= 2 {
        let quote = expression.as_bytes()[0];
        if matches!(quote, b'\'' | b'"') && expression.as_bytes().last() == Some(&quote) {
            return Some(CompileTimeValue::String(
                expression[1..expression.len() - 1].to_owned(),
            ));
        }
    }
    None
}

fn collect_fused_type(node: Node<'_>, source: &str, context: &mut LoweringContext) {
    let Some(name) = first_identifier_node(node).and_then(|x| source.get(x.byte_range())) else {
        return;
    };
    let mut constraints = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "c_type"
            && let Some(constraint) = source.get(child.byte_range())
        {
            constraints.push(cython_type_annotation(constraint));
        }
    }
    if constraints.len() >= 2 {
        context.fused_types.push((name.to_owned(), constraints));
    }
}

fn cython_type_annotation(type_name: &str) -> String {
    let normalized = type_name
        .trim()
        .trim_start_matches("const ")
        .trim_end_matches(" const")
        .trim();
    if let Some(base) = normalized.strip_suffix('*') {
        return format!("_P[{}]", cython_type_annotation(base));
    }
    if let Some(index) = normalized.find('[') {
        let container = if normalized[index..].contains(':') {
            "_M"
        } else {
            "_P"
        };
        return format!(
            "{container}[{}]",
            cython_type_annotation(&normalized[..index])
        );
    }
    match normalized {
        "bint" => "_B".to_owned(),
        "float" => "_F".to_owned(),
        "double" | "long double" => "_D".to_owned(),
        "char" | "short" | "int" | "long" | "long long" | "signed" | "unsigned" | "signed char"
        | "unsigned char" | "signed short" | "unsigned short" | "signed int" | "unsigned int"
        | "signed long" | "unsigned long" | "signed long long" | "unsigned long long"
        | "Py_ssize_t" | "size_t" => "_I".to_owned(),
        "void" => "None".to_owned(),
        _ if normalized
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.')) =>
        {
            normalized.to_owned()
        }
        _ => "object".to_owned(),
    }
}

fn replace_keyword(node: Node<'_>, lowered: &mut [u8], replacement: &[u8]) {
    let end = node.start_byte() + replacement.len();
    lowered[node.start_byte()..end].copy_from_slice(replacement);
}

fn replace_exact(node: Node<'_>, lowered: &mut [u8], replacement: &[u8]) {
    assert_eq!(
        node.end_byte() - node.start_byte(),
        replacement.len(),
        "Cython token replacement must preserve offsets"
    );
    lowered[node.byte_range()].copy_from_slice(replacement);
}

fn replace_padded(start: usize, end: usize, replacement: &str, lowered: &mut [u8]) {
    assert!(
        replacement.len() <= end - start,
        "lowered Cython syntax must fit in its original line"
    );
    lowered[start..end].fill(b' ');
    lowered[start..start + replacement.len()].copy_from_slice(replacement.as_bytes());
}

fn replace_with_pass(node: Node<'_>, source: &str, lowered: &mut [u8]) {
    for byte in &mut lowered[node.byte_range()] {
        if *byte != b'\r' && *byte != b'\n' {
            *byte = b' ';
        }
    }
    let end = first_line_content_end(node, source);
    if end - node.start_byte() >= "pass".len() {
        lowered[node.start_byte()..node.start_byte() + "pass".len()].copy_from_slice(b"pass");
    }
}

fn line_content_end(node: Node<'_>, source: &str) -> usize {
    let bytes = source.as_bytes();
    let mut end = node.end_byte();
    while end > node.start_byte() && matches!(bytes[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    end
}

fn first_line_content_end(node: Node<'_>, source: &str) -> usize {
    source.as_bytes()[node.start_byte()..node.end_byte()]
        .iter()
        .position(|byte| matches!(byte, b'\r' | b'\n'))
        .map_or(node.end_byte(), |offset| node.start_byte() + offset)
}

fn first_named_child(node: Node<'_>) -> Option<Node<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).next()
}

fn first_descendant_of_kinds<'a>(node: Node<'a>, kinds: &[&str]) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if kinds.contains(&child.kind()) {
            return Some(child);
        }
        if let Some(descendant) = first_descendant_of_kinds(child, kinds) {
            return Some(descendant);
        }
    }
    None
}

fn first_identifier_node(node: Node<'_>) -> Option<Node<'_>> {
    immediate_identifier_nodes(node).into_iter().next()
}

fn immediate_identifier_nodes(node: Node<'_>) -> Vec<Node<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|child| child.kind() == "identifier")
        .collect()
}

#[derive(Default)]
struct CythonIndex {
    class_members: HashMap<String, BTreeMap<String, CompletionItemKind>>,
    struct_members: HashMap<String, BTreeMap<String, CompletionItemKind>>,
    var_types: HashMap<String, String>,
}

impl CythonIndex {
    fn member_map_for_base(&self, base: &str) -> Option<&BTreeMap<String, CompletionItemKind>> {
        let ty = self.var_types.get(base)?;
        self.member_map_for_type(ty)
    }

    fn member_map_for_type(&self, ty: &str) -> Option<&BTreeMap<String, CompletionItemKind>> {
        self.class_members
            .get(ty)
            .or_else(|| self.struct_members.get(ty))
    }
}

fn build_index(contents: &str) -> CythonIndex {
    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_cython::language())
        .is_err()
    {
        return CythonIndex::default();
    }
    let Some(tree) = parser.parse(contents, None) else {
        return CythonIndex::default();
    };
    let mut index = CythonIndex::default();
    collect_symbols(tree.root_node(), contents, &mut index);
    index
}

fn collect_symbols(node: Node<'_>, source: &str, index: &mut CythonIndex) {
    match node.kind() {
        "class_definition" => {
            if let Some(name) = node_text(source, node.child_by_field_name("name")) {
                let mut members = BTreeMap::new();
                if let Some(body) = node.child_by_field_name("body") {
                    collect_class_members(body, source, &mut members);
                }
                if !members.is_empty() {
                    index.class_members.insert(name, members);
                }
            }
            return;
        }
        "struct" => {
            if let Some(name) = first_identifier_child(source, node) {
                let mut members = BTreeMap::new();
                if let Some(suite) = first_child_kind(node, "struct_suite") {
                    collect_struct_members(suite, source, &mut members);
                }
                if !members.is_empty() {
                    index.struct_members.insert(name, members);
                }
            }
            return;
        }
        "cvar_def" => {
            if let Some((type_name, names)) = cvar_def_type_and_names(node, source) {
                for name in names {
                    index.var_types.insert(name, type_name.clone());
                }
            }
        }
        "cvar_decl" => {
            if let Some((type_name, names)) = cvar_decl_type_and_names(node, source) {
                for name in names {
                    index.var_types.insert(name, type_name.clone());
                }
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_symbols(child, source, index);
    }
}

fn collect_class_members(
    node: Node<'_>,
    source: &str,
    members: &mut BTreeMap<String, CompletionItemKind>,
) {
    match node.kind() {
        "cvar_def" => {
            if let Some((_type_name, names)) = cvar_def_type_and_names(node, source) {
                for name in names {
                    members.entry(name).or_insert(CompletionItemKind::FIELD);
                }
            }
        }
        "cvar_decl" => {
            for name in cvar_decl_names(node, source) {
                members.entry(name).or_insert(CompletionItemKind::FIELD);
            }
        }
        "function_definition" | "c_function_definition" => {
            if let Some(name) = node_text(source, node.child_by_field_name("name")) {
                members.entry(name).or_insert(CompletionItemKind::METHOD);
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(child.kind(), "class_definition" | "struct") {
            continue;
        }
        collect_class_members(child, source, members);
    }
}

fn collect_struct_members(
    node: Node<'_>,
    source: &str,
    members: &mut BTreeMap<String, CompletionItemKind>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "cvar_decl" {
            for name in cvar_decl_names(child, source) {
                members.entry(name).or_insert(CompletionItemKind::FIELD);
            }
        }
    }
}

fn cvar_def_type_and_names(node: Node<'_>, source: &str) -> Option<(String, Vec<String>)> {
    let mut cursor = node.walk();
    let mut maybe_typed_name = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "maybe_typed_name" {
            maybe_typed_name = Some(child);
            break;
        }
    }
    let maybe_typed_name = maybe_typed_name?;
    let type_node = maybe_typed_name.child_by_field_name("type")?;
    let type_name = node_text(source, Some(type_node))?;
    let mut names = Vec::new();
    if let Some(name_node) = maybe_typed_name.child_by_field_name("name")
        && let Some(name) = node_text(source, Some(name_node))
    {
        names.push(name);
    }
    let mut child_cursor = node.walk();
    for child in node.children(&mut child_cursor) {
        if child.kind() == "identifier"
            && let Some(name) = node_text(source, Some(child))
        {
            names.push(name);
        }
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        None
    } else {
        Some((type_name, names))
    }
}

fn cvar_decl_names(node: Node<'_>, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier"
            && let Some(name) = node_text(source, Some(child))
        {
            names.push(name);
        }
    }
    names.sort();
    names.dedup();
    names
}

fn cvar_decl_type_and_names(node: Node<'_>, source: &str) -> Option<(String, Vec<String>)> {
    let mut cursor = node.walk();
    let mut type_node = None;
    for child in node.children(&mut cursor) {
        if child.kind() == "c_type" {
            type_node = Some(child);
            break;
        }
    }
    let type_name = node_text(source, type_node)?;
    let mut names = cvar_decl_names(node, source);
    names.retain(|name| name != &type_name);
    if names.is_empty() {
        None
    } else {
        Some((type_name, names))
    }
}

fn attribute_base_at(contents: &str, position: TextSize) -> Option<String> {
    let mut pos = position.to_usize().min(contents.len());
    let bytes = contents.as_bytes();
    if pos == 0 {
        return None;
    }
    if pos < contents.len() && bytes[pos] == b'.' {
        pos += 1;
    }
    let mut end = pos;
    while end > 0 && is_ident_char(bytes[end - 1]) {
        end -= 1;
    }
    if end == 0 || bytes[end - 1] != b'.' {
        return None;
    }
    let mut base_end = end - 1;
    while base_end > 0 && bytes[base_end - 1].is_ascii_whitespace() {
        base_end -= 1;
    }
    let mut base_start = base_end;
    while base_start > 0 && is_ident_char(bytes[base_start - 1]) {
        base_start -= 1;
    }
    if base_start == base_end {
        return None;
    }
    Some(contents[base_start..base_end].to_owned())
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn keyword_completion_items() -> Vec<CompletionItem> {
    const KEYWORDS: &[&str] = &[
        "cdef", "cpdef", "cimport", "ctypedef", "cclass", "nogil", "gil", "inline", "extern",
        "public", "api", "readonly", "fused", "except", "noexcept", "struct", "union", "cppclass",
        "enum",
    ];
    KEYWORDS
        .iter()
        .map(|keyword| CompletionItem {
            label: (*keyword).to_owned(),
            kind: Some(CompletionItemKind::KEYWORD),
            ..Default::default()
        })
        .collect()
}

fn node_text(source: &str, node: Option<Node<'_>>) -> Option<String> {
    let node = node?;
    let start = node.start_byte();
    let end = node.end_byte();
    source.get(start..end).map(|s| s.to_owned())
}

fn first_identifier_child(source: &str, node: Node<'_>) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            return node_text(source, Some(child));
        }
    }
    None
}

fn first_child_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_cython_declarations_to_python() {
        let source = r#"cdef struct Point:
    int x

cdef class Box:
    cdef int value

cdef Point point
"#;
        let lowered = lower_to_python(source);
        let lowered = lowered
            .source
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            lowered,
            r#"class       Point:
    x:_I

class      Box:
    value:_I

point:Point"#
        );
    }

    #[test]
    fn lower_cython_shadow_types_and_compile_time_branches() {
        let source = r#"DEF ENABLED = False
IF ENABLED:
    cdef int* disabled
ELSE:
    cdef int[:] enabled
"#;
        let lowered = lower_to_python(source);
        let lowered = lowered
            .source
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            lowered,
            r#"ENABLED=False
if ENABLED:
    pass
else:
    enabled:_M[_I]"#
        );
    }

    #[test]
    fn lower_fused_type_to_type_var_prelude() {
        let source = r#"ctypedef fused number:
    int
    double
"#;
        let lowered = lower_to_python(source);
        assert!(lowered.source.starts_with("pass"));
        assert!(lowered.prelude.contains(r#"number = _V("number", _I, _D)"#));
    }
}
