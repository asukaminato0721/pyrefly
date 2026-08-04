/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::path::PathBuf;

const DEFAULT_ENVIRONMENT: &str = ".pixi/envs/default";

#[cfg(windows)]
fn interpreter(root: &Path) -> PathBuf {
    root.join("python.exe")
}

#[cfg(not(windows))]
fn interpreter(root: &Path) -> PathBuf {
    root.join("bin/python")
}

/// Find the default Pixi environment for the workspace containing `project_path`.
pub fn find(project_path: &Path) -> Option<PathBuf> {
    project_path
        .ancestors()
        .map(|root| interpreter(&root.join(DEFAULT_ENVIRONMENT)))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use pyrefly_util::test_path::TestPath;

    use super::*;

    #[cfg(windows)]
    const INTERPRETER: &str = ".pixi/envs/default/python.exe";
    #[cfg(not(windows))]
    const INTERPRETER: &str = ".pixi/envs/default/bin/python";

    #[cfg(windows)]
    fn default_environment() -> TestPath {
        TestPath::dir("default", vec![TestPath::file("python.exe")])
    }

    #[cfg(not(windows))]
    fn default_environment() -> TestPath {
        TestPath::dir(
            "default",
            vec![TestPath::dir("bin", vec![TestPath::file("python")])],
        )
    }

    #[test]
    fn test_find_default_environment() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(
            root,
            vec![
                TestPath::dir(
                    ".pixi",
                    vec![TestPath::dir("envs", vec![default_environment()])],
                ),
                TestPath::dir("src", vec![TestPath::file("main.py")]),
            ],
        );

        assert_eq!(find(&root.join("src")), Some(root.join(INTERPRETER)));
    }

    #[test]
    fn test_find_no_default_environment() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        TestPath::setup_test_directory(root, vec![TestPath::file("pixi.toml")]);

        assert_eq!(find(root), None);
    }
}
