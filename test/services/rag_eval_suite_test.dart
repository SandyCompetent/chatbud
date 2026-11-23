import 'dart:convert';
import 'dart:io';
import 'package:chatbud/src/models/rag_citation.dart';
import 'package:flutter_test/flutter_test.dart';

/// Sanity checks for the golden evaluation dataset (`eval/data/golden_eval_dataset.json`).
///
/// These tests validate the DATASET only. They do not measure the quality of
/// ChatBud's model or retrieval. Model scoring lives in `eval/run_evals.py`
/// (run with `--responses` on real model output from `eval/generate_responses.py`).
void main() {
  group('Golden evaluation dataset sanity checks', () {
    late Map<String, dynamic> evalDataset;
    late List<dynamic> queries;

    final numberPattern = RegExp(r'\d[\d,]*(?:\.\d+)?');

    Set<double> numbersIn(String text) => numberPattern
        .allMatches(text)
        .map((m) => double.tryParse(m.group(0)!.replaceAll(',', '')))
        .whereType<double>()
        .toSet();

    setUpAll(() {
      final file = File('eval/data/golden_eval_dataset.json');
      expect(file.existsSync(), isTrue,
          reason: 'Golden eval dataset file must exist at eval/data/golden_eval_dataset.json');
      final content = file.readAsStringSync();
      evalDataset = jsonDecode(content) as Map<String, dynamic>;
      queries = evalDataset['queries'] as List<dynamic>;
    });

    test('validates golden dataset schema and complete 100-query coverage', () {
      expect(queries.length, equals(100));

      final categoryCounts = <String, int>{};
      for (final q in queries) {
        final map = q as Map<String, dynamic>;
        expect(map['id'], isNotEmpty);
        expect(map['category'], isNotEmpty);
        expect(map['query'], isNotEmpty);
        expect(map['ground_truth_context'], isNotEmpty);
        expect(map['reference_answer'], isNotEmpty);

        final cat = map['category'] as String;
        categoryCounts[cat] = (categoryCounts[cat] ?? 0) + 1;
      }

      // 4 balanced domains with 25 queries each
      expect(categoryCounts['live_public_api'], equals(25));
      expect(categoryCounts['exeter_academic'], equals(25));
      expect(categoryCounts['memory_vault'], equals(25));
      expect(categoryCounts['persona_specialist'], equals(25));
    });

    test('serializes golden ground-truth context into RagGroundingResult prompt blocks', () {
      final sampleItem = queries.firstWhere(
        (q) => (q as Map<String, dynamic>)['id'] == 'eval_api_001',
      ) as Map<String, dynamic>;

      final contextSnippets = (sampleItem['ground_truth_context'] as List<dynamic>)
          .map((e) => e.toString())
          .toList();

      final citations = contextSnippets.asMap().entries.map((entry) {
        return RagCitation(
          sourceName: 'Frankfurter ECB Currency API',
          title: 'EUR to USD Reference Rate',
          url: 'https://api.frankfurter.app',
          snippet: entry.value,
          category: 'Finance',
        );
      }).toList();

      final ragResult = RagGroundingResult(
        personaId: sampleItem['persona_id'] as String,
        query: sampleItem['query'] as String,
        citations: citations,
        retrievedAt: DateTime(2026, 10, 24),
      );

      expect(ragResult.hasContext, isTrue);
      expect(ragResult.citations.length, equals(contextSnippets.length));

      final promptBlock = ragResult.toSystemPromptBlock();
      expect(promptBlock, contains('=== LIVE GROUNDED RAG DATA (VERIFIED PUBLIC APIs) ==='));
      expect(promptBlock, contains('Frankfurter API European Central Bank reference rate'));
      expect(promptBlock, contains('URL: https://api.frankfurter.app'));
    });

    test('query ids are unique', () {
      final ids = queries.map((q) => (q as Map<String, dynamic>)['id'] as String).toList();
      expect(ids.toSet().length, equals(ids.length), reason: 'Duplicate query ids found');
    });

    // Crude lexical guard: catches a reference answer that is unrelated to its
    // context (< 10% shared words). It cannot judge paraphrase. At a 25% bar,
    // eval_mem_012, eval_mem_014, eval_per_003 and eval_per_023 are flagged,
    // but they were reviewed by hand and are valid paraphrases of their context.
    test('no reference answer is unrelated to its own ground-truth context', () {
      final weak = <String>[];
      for (final item in queries) {
        final map = item as Map<String, dynamic>;
        final answer = (map['reference_answer'] as String).toLowerCase();
        final context = (map['ground_truth_context'] as List<dynamic>)
            .map((e) => e.toString().toLowerCase())
            .join(' ');

        final words = answer
            .replaceAll(RegExp(r'[^\w\s\d]'), ' ')
            .split(' ')
            .where((w) => w.length > 3)
            .toList();
        if (words.isEmpty) continue;

        final matched = words.where((w) => context.contains(w)).length;
        if (matched / words.length < 0.10) {
          weak.add(map['id'] as String);
        }
      }
      expect(weak, isEmpty,
          reason: 'Reference answers barely overlapping their context: $weak');
    });

    test('every number in a reference answer also appears in its ground-truth context', () {
      final unsupported = <String>[];
      for (final item in queries) {
        final map = item as Map<String, dynamic>;
        final context = (map['ground_truth_context'] as List<dynamic>)
            .map((e) => e.toString())
            .join(' ');
        final answerNumbers = numbersIn(map['reference_answer'] as String);
        if (!numbersIn(context).containsAll(answerNumbers)) {
          unsupported.add(map['id'] as String);
        }
      }
      expect(unsupported, isEmpty,
          reason: 'Reference answers containing numbers not in their context: $unsupported');
    });
  });
}
