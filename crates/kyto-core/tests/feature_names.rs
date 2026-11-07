//! Task 3.4: the syntax of `feature(...)` names. Short and full names parse, `Display`
//! gives the parsed text back, names that can never resolve are `KY-E208`, and
//! `CellExpr::feature` leaves an unresolved `Source::Feature` leaf (resolution is 3.5).
#![allow(clippy::unwrap_used)]

use kyto_core::{BinaryOp, CellExpr, ExprDtype, ExprNode, FeatureName, Literal, Source};
use kyto_error::KytoError;
use proptest::prelude::*;

fn short(name: &str) -> FeatureName {
    FeatureName::Short {
        name: name.to_owned(),
    }
}

fn full(prefix: &str, name: &str) -> FeatureName {
    FeatureName::Full {
        prefix: prefix.to_owned(),
        name: name.to_owned(),
    }
}

#[test]
fn short_and_full_names_parse() {
    assert_eq!(FeatureName::parse("CD3E").unwrap(), short("CD3E"));
    assert_eq!(FeatureName::parse("cluster").unwrap(), short("cluster"));
    assert_eq!(
        FeatureName::parse("log1p:CD3E").unwrap(),
        full("log1p", "CD3E")
    );
    assert_eq!(
        FeatureName::parse("meta:cluster").unwrap(),
        full("meta", "cluster")
    );
    // Split at the first ':' only: the name part may hold more.
    assert_eq!(
        FeatureName::parse("counts:HLA:DRA").unwrap(),
        full("counts", "HLA:DRA")
    );
    assert_eq!(FeatureName::parse("pca:0").unwrap(), full("pca", "0"));
    assert_eq!(
        FeatureName::parse("raw.v-2_x:A").unwrap(),
        full("raw.v-2_x", "A")
    );
    // A short name is kept as written; spaces and case are the resolver's business.
    assert_eq!(FeatureName::parse(" cd3e").unwrap(), short(" cd3e"));
}

fn problem(text: &str) -> String {
    match FeatureName::parse(text).unwrap_err() {
        KytoError::FeatureNameSyntax { name, problem } => {
            assert_eq!(name, text);
            problem
        }
        other => panic!("expected KY-E208, got {other:?}"),
    }
}

#[test]
fn names_that_cannot_resolve_are_ky_e208() {
    assert_eq!(problem(""), "it is empty");
    assert_eq!(problem(":CD3E"), "nothing comes before ':'");
    assert_eq!(problem(":"), "nothing comes before ':'");
    assert_eq!(problem("log1p:"), "nothing follows 'log1p:'");
    assert!(problem("log 1p:CD3E").contains("the prefix 'log 1p'"));
    assert!(problem("é:CD3E").contains("names match [A-Za-z0-9_.-]+"));
    let error = FeatureName::parse("log1p:").unwrap_err();
    assert_eq!(error.code(), "KY-E208");
    assert_eq!(
        error.to_string(),
        "KY-E208: invalid feature name 'log1p:' in feature(...): nothing follows 'log1p:'"
    );
    assert!(error.hint().contains("'log1p:CD3E'"));
}

#[test]
fn feature_leaves_an_unresolved_leaf() {
    let expr = CellExpr::feature("log1p:CD3E").unwrap();
    assert_eq!(expr.dtype(), ExprDtype::Unknown);
    assert_eq!(
        **expr.node(),
        ExprNode::Source(Source::Feature {
            name: full("log1p", "CD3E")
        })
    );
    assert!(!expr.is_metadata_only());
    assert!(expr.to_polars().is_none());
    assert_eq!(expr.to_string(), "feature('log1p:CD3E')");
    let compared = CellExpr::binary(expr, BinaryOp::Gt, Literal::Float(1.5)).unwrap();
    assert_eq!(compared.to_string(), "feature('log1p:CD3E') > 1.5");
    assert_eq!(
        CellExpr::feature("it's").unwrap().to_string(),
        r"feature('it\'s')"
    );
    assert_eq!(CellExpr::feature("").unwrap_err().code(), "KY-E208");
}

proptest! {
    /// Any name that parses renders back to the text it was parsed from.
    #[test]
    fn display_gives_the_parsed_text(text in "\\PC{0,12}(:\\PC{0,12}){0,2}") {
        if let Ok(name) = FeatureName::parse(&text) {
            prop_assert_eq!(name.to_string(), text.clone());
            prop_assert_eq!(FeatureName::parse(&name.to_string()).unwrap(), name);
        }
    }
}
