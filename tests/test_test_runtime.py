def test_pytest_uses_sys_capture_for_tk_runtime(pytestconfig):
    assert pytestconfig.getoption("capture") == "sys"
