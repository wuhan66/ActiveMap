from scripts.assess_active_catalog_rl_readiness import assess


def test_rl_requires_all_three_seed_map_and_safety_gates():
    selector = {
        "schema_version": "active-catalog-paper-promotion-v1",
        "passed": True, "seed_count": 3, "test_assets_read": False,
    }
    branch = {
        "schema_version": "active-catalog-tool-branch-promotion-v1",
        "promote": True, "minimum_seed_count": 3, "test_assets_read": False,
    }
    writeback = {
        "schema_version": "active-catalog-tool-writeback-promotion-v1",
        "promote": True, "minimum_seed_count": 3, "test_assets_read": False,
    }
    assert assess(selector, branch, writeback)["ready_for_single_seed_rl"] is True
    writeback["promote"] = False
    assert assess(selector, branch, writeback)["ready_for_single_seed_rl"] is False
