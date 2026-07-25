from aai_core.diagnostics import _module_available


def test_missing_nested_module_is_reported_without_crashing():
    assert not _module_available("definitely_missing_parent.child")
