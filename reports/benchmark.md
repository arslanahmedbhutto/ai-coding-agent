# Benchmark results

8 tasks, each graded by hidden pytest tests the agent never sees.

| Model | Passed | Avg tool calls | Avg time (s) | Malformed calls retried | Loops broken | Run errors |
|---|---|---|---|---|---|---|
| openai/gpt-oss-120b | 8/8 (100%) | 5.8 | 28.1 | 3 | 5 | 0 |
| openai/gpt-oss-20b | 8/8 (100%) | 5.9 | 41.8 | 0 | 2 | 0 |
| qwen/qwen3.8-27b | 8/8 (100%) | 7.4 | 155.8 | 0 | 14 | 2 |

## Per task

| Task | openai/gpt-oss-120b | openai/gpt-oss-20b | qwen/qwen3.8-27b |
|---|---|---|---|
| calculator | ✅ | ✅ | ✅ |
| csv_summary | ✅ | ✅ | ✅ |
| fix_average | ✅ | ✅ | ✅ |
| fix_discount | ✅ | ✅ | ✅ |
| fizzbuzz | ✅ | ✅ | ✅ |
| primes | ✅ | ✅ | ✅ |
| stack | ✅ | ✅ | ✅ |
| word_count | ✅ | ✅ | ✅ |
