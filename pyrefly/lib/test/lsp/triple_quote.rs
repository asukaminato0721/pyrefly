/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::test::util::TestEnv;

fn should_auto_close(code: &str, position: usize, quote: &str) -> bool {
    let mut env = TestEnv::new();
    env.add("main", code);
    let (state, handle_for) = env.to_state();
    let handle: Handle = handle_for("main");
    state.transaction().should_auto_close_triple_quote(
        &handle,
        TextSize::new(position as u32),
        quote,
    )
}

#[test]
fn auto_close_triple_quote_openers() {
    for (code, quote) in [
        ("\"\"", "\""),
        ("''", "'"),
        ("r\"\"", "\""),
        ("f\"\"", "\""),
        ("def foo():\n    \"\"", "\""),
        ("value = b''", "'"),
        ("value = \"previous\" \"\"", "\""),
    ] {
        assert!(
            should_auto_close(code, code.len(), quote),
            "expected an opener for {code:?}"
        );
    }
}

#[test]
fn do_not_auto_close_non_openers() {
    for (code, position, quote) in [
        ("\"\"\"content\"\"", 12, "\""),
        ("'''content''", 12, "'"),
        ("\"\"\"docstring\"\"\"", 14, "\""),
        ("# \"\"", 4, "\""),
        ("\"\"", 2, "'"),
    ] {
        assert!(
            !should_auto_close(code, position, quote),
            "unexpected opener for {code:?} at {position}"
        );
    }
}
