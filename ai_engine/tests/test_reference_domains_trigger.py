from ai_engine.orchestrator.reference_domains_trigger import is_domain_reference_query


def test_medical_keywords():
    assert is_domain_reference_query("What are the symptoms of diabetes and hypertension?") is True


def test_psych_keywords():
    assert is_domain_reference_query("I have anxiety and depression; what therapy exists?") is True


def test_anatomy():
    assert is_domain_reference_query("Where is the liver relative to the stomach in human anatomy?") is True


def test_short_or_irrelevant():
    assert is_domain_reference_query("hi") is False
    assert is_domain_reference_query("what is the weather") is False
