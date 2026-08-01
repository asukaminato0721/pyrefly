/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::TextDocumentPositionParams;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct AutoCloseTripleQuoteParams {
    #[serde(flatten)]
    pub text_document_position: TextDocumentPositionParams,
    pub quote: String,
}

pub(crate) enum AutoCloseTripleQuoteRequest {}

impl lsp_types::request::Request for AutoCloseTripleQuoteRequest {
    type Params = AutoCloseTripleQuoteParams;
    type Result = bool;
    const METHOD: &'static str = "pyrefly/textDocument/autoCloseTripleQuote";
}
