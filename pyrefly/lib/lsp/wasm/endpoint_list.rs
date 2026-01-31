/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use lsp_types::Location;
use lsp_types::request::Request;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug)]
pub enum EndpointList {}

impl Request for EndpointList {
    type Params = EndpointListParams;
    type Result = Vec<EndpointItem>;
    const METHOD: &'static str = "pyrefly/endpoints";
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EndpointListParams {}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EndpointItem {
    pub path: String,
    pub methods: Vec<String>,
    pub location: Location,
}
