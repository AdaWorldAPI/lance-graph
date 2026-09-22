mod common;
use common::*;
use lance_graph_mask_risc::{reference_execute, Planes, Value};
use lance_graph_sap::query::CatsQuery;

#[test]
fn employee_date_activity_sum_matches_independent_oracle_across_word_tails() {
    for n in [0, 1, 63, 64, 65, 131, 4097] {
        let mut input = fixture(n);
        let mut expected_dev = 0;
        let mut expected_ops = 0;
        for i in 0..n {
            input[5][i] = Some(if i % 3 == 0 { "00000007" } else { "00000042" });
            input[9][i] = Some(if i % 5 == 0 {
                "2026-08-31T23:59:59Z"
            } else {
                "2026-09-01T12:34:56Z"
            });
            input[11][i] = Some(if i % 2 == 0 { "DEV" } else { "OPS" });
            input[10][i] = Some(if i % 2 == 0 { "8.50" } else { "0.125" });
            if i % 3 != 0 && i % 5 != 0 {
                if i % 2 == 0 {
                    expected_dev += 8500;
                } else {
                    expected_ops += 125;
                }
            }
        }
        let batch = bind(&input);
        let mut query = CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30").unwrap();
        let mut sums = vec![0; query.groups()];
        let mask = query.execute_into(&mut sums).unwrap().to_vec(); // TEST-only copy for oracle
        for (i, &sum) in sums.iter().enumerate() {
            let expected = match batch.activity_label(i as u32) {
                Some("DEV") => expected_dev,
                Some("OPS") => expected_ops,
                None => 0,
                _ => unreachable!(),
            };
            assert_eq!(sum, expected, "n={n}, group={i}");
        }
        let lanes = batch.lanes();
        let masks = [mask.as_slice()];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        for (program, sum) in query.plan().groups.iter().zip(sums) {
            assert_eq!(
                reference_execute(program, &planes, None).unwrap(),
                Value::SumI64(sum)
            );
        }
        let mut absent =
            CatsQuery::prepare(&batch, "99999999", "2026-09-01", "2026-09-30").unwrap();
        let mut sums = vec![1; absent.groups()];
        absent.execute_into(&mut sums).unwrap();
        assert!(sums.iter().all(|v| *v == 0));
    }
}
