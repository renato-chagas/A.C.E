"""Tests for knowledge graph storage backend (Phase 15.3)."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from openjarvis.tools.storage.knowledge_graph import (
    Entity,
    KnowledgeGraphMemory,
    Relation,
    stable_relation_key,
)


class TestKnowledgeGraph:
    def _make_kg(self, tmp_path):
        return KnowledgeGraphMemory(db_path=tmp_path / "kg.db")

    def test_add_and_get_entity(self, tmp_path):
        kg = self._make_kg(tmp_path)
        entity = Entity(
            entity_id="e1",
            entity_type="concept",
            name="Machine Learning",
            properties={"field": "AI"},
        )
        kg.add_entity(entity)
        result = kg.get_entity("e1")
        assert result is not None
        assert result.name == "Machine Learning"
        assert result.properties["field"] == "AI"
        kg.close()

    def test_entity_not_found(self, tmp_path):
        kg = self._make_kg(tmp_path)
        assert kg.get_entity("nonexistent") is None
        kg.close()

    def test_count_episode_links_matches_full_ids_only(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="concept", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="concept", name="B"))
        episode_id = "a" * 32
        kg.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type="mentions",
                properties={"episode_ids": [episode_id]},
            )
        )

        assert kg.count_episode_links(episode_id.upper()) == 1
        assert kg.count_episode_links("b" * 32) == 0
        assert kg.count_episode_links("short") == 0
        kg.close()

    def test_purge_episode_data_preserves_independent_sources(self, tmp_path):
        kg = self._make_kg(tmp_path)
        episode_id = "a" * 32
        retained_episode_id = "b" * 32
        kg.add_entity(
            Entity(
                "project",
                "project",
                "ACE",
                source="conversation",
                properties={
                    "evidence_quote": "ACE has a private decision.",
                    "evidence_sources": [
                        {
                            "source": "conversation:chat:turn-a",
                            "source_kind": "conversation",
                            "episode_id": episode_id,
                            "quote": "ACE has a private decision.",
                        },
                        {
                            "source": "project-notes.md",
                            "source_kind": "document",
                            "quote": "ACE is an assistant project.",
                        },
                    ],
                },
            )
        )
        kg.add_entity(Entity("framework", "software", "OpenJarvis"))
        kg.add_relation(
            Relation(
                "project",
                "framework",
                "built_on",
                properties={"episode_ids": [episode_id, retained_episode_id]},
                evidence=[
                    {
                        "source": "conversation:chat:turn-a",
                        "quote": "ACE is built on OpenJarvis.",
                        "episode_id": episode_id,
                    },
                    {
                        "source": "web",
                        "source_url": "https://example.com/ace",
                        "quote": "OpenJarvis is a software framework.",
                    },
                ],
            )
        )

        counts = kg.purge_episode_data([episode_id])
        relation = kg.query_pattern(relation_type="built_on").relations[0]
        project = kg.get_entity("project")
        assert counts["evidence_removed"] == 2
        assert counts["relations_updated"] == 1
        assert relation.properties["episode_ids"] == [retained_episode_id]
        assert [item["source"] for item in relation.evidence] == ["web"]
        assert project is not None
        assert project.source == "document"
        assert project.properties["evidence_quote"] == "ACE is an assistant project."
        serialized_events = json.dumps(kg.recent_events(limit=50), ensure_ascii=False)
        assert "private decision" not in serialized_events
        assert "ACE is built on OpenJarvis." not in serialized_events

        clear_counts = kg.purge_episode_data(None)
        relation = kg.query_pattern(relation_type="built_on").relations[0]
        assert clear_counts["relations_updated"] == 1
        assert "episode_ids" not in relation.properties
        assert [item["source"] for item in relation.evidence] == ["web"]
        assert kg.recent_events(limit=1)[0]["event_type"] == "memory_data_purged"
        kg.close()

    def test_add_relation(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="concept", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="concept", name="B"))
        kg.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type="depends_on",
                evidence=[{"source": "manual", "quote": "A depends on B."}],
            )
        )
        assert kg.relation_count() == 1
        event = kg.recent_events(limit=1)[0]
        assert event["event_type"] == "relation_added"
        assert event["after"]["evidence"][0]["quote"] == "A depends on B."
        kg.close()

    def test_recent_events_support_stable_descending_pagination(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity("first", "concept", "First"))
        kg.add_entity(Entity("second", "concept", "Second"))
        kg.add_entity(Entity("first", "concept", "First renamed"))

        first_page = kg.recent_events(limit=2)
        assert [event["event_type"] for event in first_page] == [
            "entity_updated",
            "entity_added",
        ]
        second_page = kg.recent_events(limit=2, before_id=first_page[-1]["id"])
        assert [event["event_type"] for event in second_page] == ["entity_added"]
        assert second_page[0]["id"] < first_page[-1]["id"]
        assert kg.recent_events(limit=2, before_id=second_page[0]["id"]) == []
        kg.close()

    def test_invalid_evidence_rolls_back_relation_and_audit_event(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity("first", "concept", "First"))
        kg.add_entity(Entity("second", "concept", "Second"))
        event_count = len(kg.recent_events())

        with pytest.raises(ValueError, match="valid_from"):
            kg.add_relation(
                Relation(
                    "first",
                    "second",
                    "depends_on",
                    evidence=[
                        {
                            "source": "manual",
                            "quote": "The dates are invalid.",
                            "valid_from": "2026-02-30",
                        }
                    ],
                )
            )

        assert kg.relation_count() == 0
        assert len(kg.recent_events()) == event_count
        kg.close()

    def test_neighbors(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="concept", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="concept", name="B"))
        kg.add_entity(Entity(entity_id="c", entity_type="concept", name="C"))
        kg.add_relation(Relation(source_id="a", target_id="b", relation_type="uses"))
        kg.add_relation(Relation(source_id="a", target_id="c", relation_type="uses"))
        neighbors = kg.neighbors("a", direction="out")
        assert len(neighbors) == 2
        names = {n.name for n in neighbors}
        assert names == {"B", "C"}
        kg.close()

    def test_neighbors_with_type_filter(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="concept", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="concept", name="B"))
        kg.add_entity(Entity(entity_id="c", entity_type="concept", name="C"))
        kg.add_relation(Relation(source_id="a", target_id="b", relation_type="uses"))
        kg.add_relation(
            Relation(
                source_id="a",
                target_id="c",
                relation_type="produces",
            )
        )
        neighbors = kg.neighbors("a", relation_type="uses", direction="out")
        assert len(neighbors) == 1
        assert neighbors[0].name == "B"
        kg.close()

    def test_query_pattern_entities(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="t1", entity_type="tool", name="Calculator"))
        kg.add_entity(Entity(entity_id="t2", entity_type="tool", name="Search"))
        kg.add_entity(Entity(entity_id="a1", entity_type="agent", name="Bot"))
        result = kg.query_pattern(entity_type="tool")
        assert len(result.entities) == 2
        kg.close()

    def test_query_pattern_relations(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="x", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="x", name="B"))
        kg.add_relation(Relation(source_id="a", target_id="b", relation_type="used"))
        result = kg.query_pattern(relation_type="used")
        assert len(result.relations) == 1
        kg.close()

    def test_memory_backend_store_retrieve(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.store("doc1", "hello world", metadata={"name": "greeting"})
        content = kg.retrieve("doc1")
        assert content is not None
        assert "hello world" in content
        kg.close()

    def test_memory_backend_search(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.store("doc1", "machine learning", metadata={"name": "ML"})
        kg.store("doc2", "deep learning", metadata={"name": "DL"})
        results = kg.search("learning")
        assert len(results) >= 1
        kg.close()

    def test_delete(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(
            Entity(
                entity_id="x",
                entity_type="test",
                name="Private project title",
                properties={"evidence_quote": "Private project sentence."},
            )
        )
        kg.add_entity(Entity("y", "software", "OpenJarvis"))
        kg.add_relation(
            Relation(
                "x",
                "y",
                "uses",
                evidence=[{"source": "manual", "quote": "Private relation quote."}],
            )
        )
        assert kg.delete("x")
        assert kg.get_entity("x") is None
        events = kg.recent_events(limit=50)
        serialized = json.dumps(events, ensure_ascii=False)
        assert "Private project title" not in serialized
        assert "Private project sentence" not in serialized
        assert "Private relation quote" not in serialized
        deletion = next(
            event for event in events if event["event_type"] == "entity_deleted"
        )
        assert deletion["before"] == {
            "details_purged": True,
            "removed_relation_count": 1,
        }
        kg.close()

    def test_clear(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="test", name="A"))
        kg.add_entity(Entity(entity_id="b", entity_type="test", name="B"))
        kg.clear()
        assert kg.entity_count() == 0
        events = kg.recent_events(limit=50)
        assert [event["event_type"] for event in events] == ["graph_cleared"]
        assert events[0]["before"] == {"entities": 2, "relations": 0}
        kg.close()

    def test_legacy_delete_history_is_scrubbed_on_open(self, tmp_path):
        path = tmp_path / "legacy-delete-history.db"
        kg = KnowledgeGraphMemory(db_path=path)
        kg.add_entity(Entity("private", "project", "Old private name"))
        kg.add_entity(Entity("public", "software", "OpenJarvis"))
        kg.add_relation(
            Relation(
                "private",
                "public",
                "uses",
                evidence=[{"source": "manual", "quote": "Old private quote."}],
            )
        )
        relation_id = kg._conn.execute("SELECT id FROM relations").fetchone()["id"]
        kg.close()

        connection = sqlite3.connect(path)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO graph_events "
            "(event_type, entity_id, relation_id, source, before_json, after_json, "
            "created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "entity_deleted",
                "private",
                0,
                "manual",
                json.dumps(
                    {
                        "entity": {"entity_id": "private", "name": "Old private name"},
                        "relations": [
                            {
                                "id": relation_id,
                                "source_id": "private",
                                "target_id": "public",
                                "evidence": [{"quote": "Old private quote."}],
                            }
                        ],
                    }
                ),
                "{}",
                time.time(),
            ),
        )
        connection.execute("DELETE FROM relations WHERE id=?", (relation_id,))
        connection.execute("DELETE FROM entities WHERE entity_id='private'")
        connection.commit()
        connection.close()

        migrated = KnowledgeGraphMemory(path)
        events = migrated.recent_events(limit=50)
        deletion = next(
            event for event in events if event["event_type"] == "entity_deleted"
        )
        assert deletion["before"] == {
            "details_purged": True,
            "removed_relation_count": 1,
        }
        assert all(event["entity_id"] != "private" for event in events)
        assert "Old private" not in json.dumps(events, ensure_ascii=False)
        migrated.close()

    def test_entity_count(self, tmp_path):
        kg = self._make_kg(tmp_path)
        assert kg.entity_count() == 0
        kg.add_entity(Entity(entity_id="a", entity_type="test", name="A"))
        assert kg.entity_count() == 1
        kg.close()

    def test_relation_dedup_accumulates_provenance_and_survives_reopen(self, tmp_path):
        path = tmp_path / "provenance.db"
        kg = KnowledgeGraphMemory(db_path=path)
        kg.add_entity(Entity(entity_id="a", entity_type="company", name="Acme"))
        kg.add_entity(Entity(entity_id="b", entity_type="product", name="Quartz"))
        first = Relation(
            source_id="a",
            target_id="b",
            relation_type="develops",
            source="web",
            source_url="https://one.example/",
            evidence=[
                {
                    "source": "web",
                    "source_url": "https://one.example/",
                    "quote": "Acme develops Quartz.",
                }
            ],
        )
        kg.add_relation(first)
        kg.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type="DEVELOPS",
                source="web",
                source_url="https://two.example/",
                evidence=[
                    {
                        "source": "web",
                        "source_url": "https://two.example/",
                        "quote": "Quartz is developed by Acme.",
                    }
                ],
            )
        )
        assert kg.relation_count() == 1
        assert kg.evidence_count() == 2
        relation_key = (
            kg.query_pattern(relation_type="develops").relations[0].relation_id
        )
        kg.close()

        reopened = KnowledgeGraphMemory(db_path=path)
        relation = reopened.relation_by_key(relation_key)
        assert relation is not None
        assert {item["source_url"] for item in relation.evidence} == {
            "https://one.example/",
            "https://two.example/",
        }
        assert "Acme" in reopened.context_for("Acme")
        assert reopened.delete_relation(relation_key)
        assert reopened.relation_count() == 0
        assert reopened.evidence_count() == 0
        reopened.close()

    def test_legacy_duplicate_relations_keep_all_provenance(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY, entity_type TEXT, name TEXT,
                properties TEXT DEFAULT '{}', created_at REAL DEFAULT 0
            );
            CREATE TABLE relations (
                id INTEGER PRIMARY KEY, source_id TEXT, target_id TEXT,
                relation_type TEXT, weight REAL DEFAULT 1,
                properties TEXT DEFAULT '{}', created_at REAL DEFAULT 0
            );
            """
        )
        conn.executemany(
            "INSERT INTO entities(entity_id, entity_type, name) VALUES (?, ?, ?)",
            [("a", "company", "Acme"), ("b", "product", "Quartz")],
        )
        conn.executemany(
            "INSERT INTO relations(id, source_id, target_id, relation_type, "
            "properties, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "a",
                    "b",
                    "develops",
                    json.dumps(
                        {
                            "source_url": "https://one.example",
                            "evidence": "Acme makes Quartz.",
                        }
                    ),
                    1,
                ),
                (
                    2,
                    "a",
                    "b",
                    "develops",
                    json.dumps(
                        {
                            "source_url": "https://two.example",
                            "evidence": "Quartz is made by Acme.",
                        }
                    ),
                    2,
                ),
            ],
        )
        conn.commit()
        conn.close()

        kg = KnowledgeGraphMemory(db_path=path)
        assert kg.entity_count() == 2
        assert kg.relation_count() == 1
        assert kg.evidence_count() == 2
        relation = kg.query_pattern(relation_type="develops").relations[0]
        assert relation.polarity == "affirmed"
        assert relation.relation_id == stable_relation_key("a", "b", "develops")
        assert {item["source_url"] for item in relation.evidence} == {
            "https://one.example",
            "https://two.example",
        }
        kg.close()

    def test_update_relation_keeps_prior_evidence(self, tmp_path):
        kg = self._make_kg(tmp_path)
        kg.add_entity(Entity(entity_id="a", entity_type="company", name="Acme"))
        kg.add_entity(Entity(entity_id="b", entity_type="product", name="Quartz"))
        old_key = kg.add_relation(
            Relation(
                source_id="a",
                target_id="b",
                relation_type="develops",
                source="web",
                source_url="https://example.org",
                evidence=[
                    {
                        "source": "web",
                        "source_url": "https://example.org",
                        "quote": "Acme develops Quartz.",
                    }
                ],
            )
        )
        new_key = kg.update_relation(
            old_key,
            Relation(
                source_id="a",
                target_id="b",
                relation_type="manufactures",
                confidence=0.9,
                status="verified",
                source="manual",
                evidence=[{"source": "manual", "quote": "Reviewed by user."}],
            ),
        )
        assert new_key and new_key != old_key
        assert kg.relation_by_key(old_key) is None
        updated = kg.relation_by_key(new_key)
        assert updated is not None
        assert updated.status == "verified"
        assert {item["quote"] for item in updated.evidence} == {
            "Acme develops Quartz.",
            "Reviewed by user.",
        }
        assert (
            next(
                item
                for item in updated.evidence
                if item["quote"] == "Acme develops Quartz."
            )["status"]
            == "superseded"
        )
        graph_context = kg.context_for("Acme")
        assert "Reviewed by user." in graph_context
        assert "Acme develops Quartz." not in graph_context
        kg.close()

    def test_graph_context_is_bounded_and_marked_as_untrusted(self, tmp_path):
        path = tmp_path / "context.db"
        kg = KnowledgeGraphMemory(path)
        kg.add_entity(Entity("a", "company", "Acme", source_url="https://example.org"))
        kg.add_entity(Entity("b", "product", "Quartz"))
        kg.add_relation(
            Relation(
                "a",
                "b",
                "develops",
                source="web",
                source_url="https://example.org",
                evidence=[
                    {
                        "source": "web",
                        "source_url": "https://example.org",
                        "quote": "Acme develops Quartz.",
                    }
                ],
            )
        )
        kg.close()

        from openjarvis.core.types import Message, Role
        from openjarvis.tools.storage.graph_context import inject_graph_context

        messages = inject_graph_context(
            "What does Acme develop?", [Message(Role.USER, "question")], path
        )
        assert messages[0].role == Role.SYSTEM
        assert "untrusted" in messages[0].text
        assert "https://example.org" in messages[0].text
        assert messages[-1].role == Role.USER

    def test_possible_conflicts_are_candidates_and_keep_all_current_evidence(
        self, tmp_path
    ):
        kg = self._make_kg(tmp_path)
        for entity_id, name in (
            ("device", "ACE workstation"),
            ("model-a", "Qwen model A"),
            ("model-b", "Qwen model B"),
            ("model-c", "Qwen model C"),
        ):
            kg.add_entity(Entity(entity_id, "software", name))
        kg.add_relation(
            Relation(
                "device",
                "model-a",
                "current_model",
                source="conversation",
                evidence=[{"source": "conversation", "quote": "ACE uses model A."}],
            )
        )
        kg.add_relation(
            Relation(
                "device",
                "model-b",
                "CURRENT-MODEL",
                source="web",
                source_url="https://example.org/models",
                evidence=[
                    {
                        "source": "web",
                        "source_url": "https://example.org/models",
                        "quote": "ACE uses model B.",
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "device",
                "model-c",
                "current_model",
                status="superseded",
                source="legacy",
            )
        )

        candidates = kg.possible_conflicts(entity_ids=["device"])
        assert len(candidates) == 1
        assert candidates[0]["subject"].name == "ACE workstation"
        assert candidates[0]["predicate"] == "current_model"
        assert {claim["target"].name for claim in candidates[0]["claims"]} == {
            "Qwen model A",
            "Qwen model B",
        }
        assert all(
            claim["relation"].status != "superseded"
            for claim in candidates[0]["claims"]
        )

        context = kg.context_for("ACE workstation")
        assert "Possible conflict" in context
        assert "Qwen model A" in context
        assert "Qwen model B" in context
        assert "instead of choosing a winner" in context
        assert kg.relation_count() == 3
        kg.close()

    def test_validity_periods_keep_distinct_evidence_and_bound_conflict_candidates(
        self, tmp_path
    ):
        kg = self._make_kg(tmp_path)
        for entity_id, name in (
            ("device", "ACE workstation"),
            ("model-a", "Model A"),
            ("model-b", "Model B"),
        ):
            kg.add_entity(Entity(entity_id, "software", name))
        kg.add_relation(
            Relation(
                "device",
                "model-a",
                "current_model",
                evidence=[
                    {
                        "source": "source-a",
                        "quote": "The workstation used Model A.",
                        "observed_at": 100.0,
                        "valid_from": "2023-01-01",
                        "valid_until": "2024-12-31",
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "device",
                "model-b",
                "current_model",
                evidence=[
                    {
                        "source": "source-b",
                        "quote": "The workstation uses Model B.",
                        "observed_at": 150.0,
                        "valid_from": "2025-01-01",
                    }
                ],
            )
        )
        assert kg.possible_conflicts(entity_ids=["device"]) == []

        kg.add_relation(
            Relation(
                "device",
                "model-b",
                "current_model",
                evidence=[
                    {
                        "source": "source-b",
                        "quote": "The workstation uses Model B.",
                        "observed_at": 200.0,
                        "valid_from": "2024-06-01",
                        "valid_until": "2025-06-01",
                    }
                ],
            )
        )
        relation = next(
            item
            for item in kg.query_pattern(relation_type="current_model").relations
            if item.target_id == "model-b"
        )
        assert len(relation.evidence) == 2
        assert len(kg.possible_conflicts(entity_ids=["device"])) == 1
        context = kg.context_for("ACE workstation")
        assert "historical validity period: 2024-06-01 through 2025-06-01" in context
        assert "historical validity period: 2023-01-01 through 2024-12-31" in context
        kg.add_relation(
            Relation(
                "device",
                "model-b",
                "current_model",
                status="contradicted",
            )
        )
        assert kg.possible_conflicts(entity_ids=["device"]) == []
        kg.close()

    def test_validity_dates_require_ordered_iso_calendar_dates(self):
        from openjarvis.tools.storage.knowledge_graph import normalize_validity_range

        assert normalize_validity_range("2024-02-29", "2024-03-01") == (
            "2024-02-29",
            "2024-03-01",
        )
        for values in (
            ("2024-02-30", ""),
            ("2024-01-01T00:00:00Z", ""),
            ("2024-05-02", "2024-05-01"),
        ):
            try:
                normalize_validity_range(*values)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid validity range was accepted: {values}")

    def test_legacy_evidence_identity_migrates_to_preserve_dated_records(
        self, tmp_path
    ):
        path = tmp_path / "legacy-evidence.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
                name TEXT NOT NULL, properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE relations (
                id INTEGER PRIMARY KEY, source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1,
                properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE relation_evidence (
                evidence_id TEXT PRIMARY KEY, relation_id INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL DEFAULT '',
                quote TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'reported',
                observed_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (relation_id) REFERENCES relations(id) ON DELETE CASCADE,
                UNIQUE (relation_id, source, source_url, quote)
            );
            INSERT INTO entities(entity_id, entity_type, name)
                VALUES ('device', 'device', 'ACE workstation'),
                       ('model', 'software', 'ACE model');
            INSERT INTO relations(id, source_id, target_id, relation_type)
                VALUES (1, 'device', 'model', 'current_model');
            INSERT INTO relation_evidence(
                evidence_id, relation_id, source, quote, created_at
            ) VALUES ('old', 1, 'config', 'ACE uses this model.', 1.0);
            """
        )
        connection.close()

        kg = KnowledgeGraphMemory(path)
        relation_key = (
            kg.query_pattern(relation_type="current_model").relations[0].relation_id
        )
        kg.add_relation(
            Relation(
                "device",
                "model",
                "current_model",
                evidence=[
                    {
                        "source": "config",
                        "quote": "ACE uses this model.",
                        "valid_from": "2020-01-01",
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "device",
                "model",
                "current_model",
                evidence=[
                    {
                        "source": "config",
                        "quote": "ACE uses this model.",
                        "valid_from": "2021-01-01",
                    }
                ],
            )
        )
        for episode_id in ("a" * 32, "b" * 32):
            kg.add_relation(
                Relation(
                    "device",
                    "model",
                    "current_model",
                    evidence=[
                        {
                            "source": "conversation:chat:turn",
                            "quote": "ACE uses this model.",
                            "valid_from": "2020-01-01",
                            "episode_id": episode_id,
                        }
                    ],
                )
            )
        relation = kg.relation_by_key(relation_key)
        assert relation is not None
        assert len(relation.evidence) == 5
        assert {item["valid_from"] for item in relation.evidence} == {
            "",
            "2020-01-01",
            "2021-01-01",
        }
        assert {
            item["episode_id"]
            for item in relation.evidence
            if item["valid_from"] == "2020-01-01"
        } == {"", "a" * 32, "b" * 32}
        kg.close()

    def test_evidence_review_candidates_only_flag_old_mutable_claims(self, tmp_path):
        kg = self._make_kg(tmp_path)
        for entity_id, name in (
            ("device", "ACE workstation"),
            ("old-model", "Old model"),
            ("new-model", "New model"),
            ("target", "Deployment target"),
            ("project", "ACE project"),
        ):
            kg.add_entity(Entity(entity_id, "concept", name))

        as_of = time.time()
        old_observation = as_of - 40 * 86400
        recent_observation = as_of - 10 * 86400
        kg.add_relation(
            Relation(
                "device",
                "old-model",
                "current_status",
                evidence=[
                    {
                        "source": "document:old",
                        "quote": "ACE workstation uses the old model.",
                        "observed_at": as_of,
                    },
                    {
                        "source": "document:superseded",
                        "quote": (
                            "Superseded evidence must not be presented for review."
                        ),
                        "observed_at": recent_observation,
                        "status": "superseded",
                    },
                ],
            )
        )
        with kg._conn:
            kg._conn.execute(
                "UPDATE relation_evidence SET created_at=? WHERE source=?",
                (old_observation, "document:old"),
            )
        kg.add_relation(
            Relation(
                "device",
                "new-model",
                "current_status",
                evidence=[
                    {
                        "source": "document:recent",
                        "quote": "ACE workstation uses the new model.",
                        "observed_at": recent_observation,
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "device",
                "target",
                "deployment_target",
                status="superseded",
                evidence=[
                    {
                        "source": "document:old-target",
                        "quote": "ACE deploys to the old target.",
                        "observed_at": 1.0,
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "project",
                "target",
                "uses",
                evidence=[
                    {
                        "source": "document:old-static",
                        "quote": "ACE uses this target.",
                        "observed_at": 1.0,
                    }
                ],
            )
        )
        kg.add_relation(
            Relation(
                "project",
                "new-model",
                "current_version",
                evidence=[
                    {
                        "source": "document:old-observation-recently-recorded",
                        "quote": "This old source was only recently recorded.",
                        "observed_at": 1.0,
                    }
                ],
            )
        )

        candidates = kg.evidence_review_candidates(
            entity_ids=["device"], as_of=as_of, limit=1
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["relation"].target_id == "old-model"
        assert candidate["relation"].status == "reported"
        assert candidate["age_days"] == 40
        assert candidate["review_after_days"] == 30
        assert candidate["last_evidence_at"] == old_observation
        assert candidate["last_evidence"]["quote"] == (
            "ACE workstation uses the old model."
        )
        assert kg.relation_count() == 5
        context = kg.context_for("ACE workstation")
        assert "Evidence review is due" in context
        assert "40 days ago" in context
        assert "does not prove a claim is false" in context
        assert "ACE workstation uses the old model." in context
        assert "Superseded evidence must not be presented" not in context
        assert kg.evidence_review_candidates(entity_ids=["project"], as_of=as_of) == []
        kg.close()


def test_recall_links_only_visible_active_graph_evidence(tmp_path):
    with KnowledgeGraphMemory(tmp_path / "graph.db") as graph:
        graph.add_entity(Entity("notebook", "project", "Notebook"))
        graph.add_entity(Entity("archive", "system", "Archive"))
        current = "a" * 32
        inactive = "b" * 32
        graph.add_relation(
            Relation(
                "notebook",
                "archive",
                "stores_in",
                properties={"episode_ids": [current, inactive]},
                evidence=[
                    {
                        "source": "conversation:chat:current",
                        "quote": "Current archive quote",
                        "episode_id": current,
                        "observed_at": 100,
                    },
                    {
                        "source": "conversation:chat:inactive",
                        "quote": "Contradicted archive quote",
                        "episode_id": inactive,
                        "status": "contradicted",
                        "observed_at": 200,
                    },
                ],
            )
        )

        context, episode_ids = graph.context_with_episode_ids("Notebook")
        assert "Current archive quote" in context
        assert "Contradicted archive quote" not in context
        assert episode_ids == [current]

        _, truncated_ids = graph.context_with_episode_ids("Notebook", max_chars=100)
        assert truncated_ids == []

        budgeted_context, budgeted_ids = graph.context_with_episode_ids(
            "Notebook", max_words=12
        )
        assert budgeted_context == ""
        assert budgeted_ids == []
