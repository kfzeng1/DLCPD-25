from pathlib import Path

from dlcpd25_classifier.taxonomy import Taxonomy


ROOT = Path(__file__).resolve().parents[2]


def test_taxonomy_has_203_classes_and_hierarchy() -> None:
    taxonomy = Taxonomy(ROOT / "metadata" / "class-taxonomy.json")
    assert len(taxonomy.classes) == 203
    result = next(item for item in taxonomy.classes if item.official_name == "apple black rot")
    assert result.host_zh == "苹果"
    assert result.category_zh == "植物病害"


def test_garlic_is_in_disease_category() -> None:
    taxonomy = Taxonomy(ROOT / "metadata" / "class-taxonomy.json")
    result = next(item for item in taxonomy.classes if item.official_name == "garlic pest and diseases")
    assert result.category == "disease"
