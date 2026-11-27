# 離散ブラックボックス最適化

`blackbox` コレクションは
[JijZeptWorkflow](https://github.com/Jij-Inc/JijZeptWorkflow/tree/5e2766138c4a84e37fd506aebc28ee16d62cd181/benchmarks/discrete)
で作成した離散ベンチマークです。有限の候補空間、評価器、固定した入力、共有初期観測を提供します。
JijZeptWorkflowや最適化手法のパッケージへの依存はありません。

## コレクション

| 問題 | 設定数 | 候補空間・目的 | 配布方法 |
| --- | ---: | --- | --- |
| LABS | 6 | 16/32/64 bit、energyとnegative merit factor | 同梱 |
| 二次QUBO対照 | 9 | 16/32/64 bit × instance seed 0/1/2 | 同梱、OMMX変換対応 |
| Contamination | 9 | 25二値段階 × 3 seeds × 3 penalties | 同梱 |
| Pest control | 3 | 25変数・5カテゴリ、評価ごとに乱数をreset | 同梱 |
| Ising sparsification | 9 | 4×4格子の24辺選択 × 3 seeds × 3 penalties | 同梱 |
| IOH/PBO | 150 | F1–F25 × instance 1/2/3 × dimension 25/100 | `pbo` extraが必要 |
| FMIRA fitness表 | 3 | Fluorescence 13 bit、Ribozyme・Influenza_fluB 16 bit | 明示的にローカル取得 |

カタログは**189設定**です。基本パッケージだけで36設定をオフライン利用でき、150設定は
`ioh==0.3.22`、3設定は外部の表が必要です。`available_instances` は任意依存・外部取得を含む
全IDを列挙します。依存や表がインストール済みであることを示すものではありません。

## 候補の評価

評価器はすべて単一の目的値を最小化します。fitnessとPBOのnative scoreは符号を反転し、
その他は元の方向を保ちます。候補には宣言したcardinality内の有限の整数ベクトルを渡します。
不正な値は拒否されます。

```python
from ommx_quantum_benchmarks.blackbox import DiscreteBlackbox, initial_design

benchmarks = DiscreteBlackbox()
material = benchmarks("labs-16-negative-merit")
print(material.entry["cardinalities"])
print(material.evaluate([0] * 16))

# The same 20 observed candidates can initialize every compared method.
data = benchmarks.initial_data("labs-16-negative-merit", seed=0)
assert data["count"] == 20
for candidate, score in zip(data["candidates"], data["objectives"], strict=True):
    assert abs(material.evaluate(candidate) - score) < 1e-10

# Generate a different reproducible design without consulting unobserved scores.
fresh = initial_design(material, count=20, seed=7)
assert len(fresh["candidates"]) == 20
```

`Material.entry` は候補空間、instance seed、変種、出典、最適値の状態、asset・評価器のhashを
保持します。`Material.identity` はその設定を識別します。初期design seed 0/1/2はinstance seedと
optimizer seedから独立です。共有する20観測も総評価予算に含めてください。
表全体や最適値は評価器・採点側で保持し、optimizerには問い合わせ済みの観測だけを渡します。
ライブラリ・platform間のbit単位の一致は保証しません。NumPy/IOHのversionとidentityを記録してください。

## QUBOのOMMX変換

QUBO対照は通常のOMMXツールでも評価できます。

```python
qubo = benchmarks("qubo-16-s0")
instance = qubo.to_ommx()
x = [0, 1] * 8
solution = instance.evaluate({i: float(value) for i, value in enumerate(x)})
assert abs(solution.objective - qubo.evaluate(x)) < 1e-10
```

このコレクションはQOBLIBの `Instance, Solution` APIとは別です。シミュレータや表の目的値は
`evaluate` で評価し、`to_ommx()` はQUBO対照のみを扱います。
synthetic・COMBO・PBOの最適値は既知と断定しません。FMIRAの最適値は提供された有限の表の範囲で厳密です。

## 任意依存のPBO

```bash
pip install 'ommx-quantum-benchmarks[pbo]'
```

```python
pbo = benchmarks("pbo-f1-i1-d25")
assert pbo.evaluate([1] * 25) == -25.0
```

F4–F17にはW-model変換が含まれます。IOH内部のoptimum metadataを全問題の検証済み厳密最適値として
提示していません。

## FMIRAの明示的取得

このコレクションでは固定したFMIRA snapshotのリポジトリライセンスを確認できていません。
原本の表、正規化したfitness配列、初期観測を配布物から除外しています。
次の明示的な操作は指定したディレクトリへ1組の表を取得して検証します。取得したコードは実行しません。

```python
from pathlib import Path
from ommx_quantum_benchmarks.blackbox import fetch_fitness

fitness = fetch_fitness(Path("benchmark_data/fmira"), "fitness-Fluorescence")
score = fitness.evaluate([0] * 13)
local_design = initial_design(fitness, count=20, seed=0)
```

取得は再配布の許諾を与えるものではありません。行が並び替えられたRibozyme・Influenzaも、
bit列とfitnessの対応を保ってからMSB-firstのbinary code順に整列します。
値は符号のみ反転し、再正規化・補完・jitterを行いません。
Influenzaには最低値の大きなplateauがあります。測定単位、検出限界、実験上の不確かさは
このsnapshotから独立には確定していません。

## 出典と変種

- [COMBO](https://github.com/QUVA-Lab/COMBO/tree/9529eabb86365ce3a2ca44fff08291a09a853ca2)：
  BSDの通知を `blackbox/COMBO-LICENSE.txt` に保持しています。Contaminationは25段階・100シミュレーションの
  soft penalty scoreを保持し、hard constraintには変換しません。
  Pestは評価ごとにlocal RandomStateをresetする決定的な変種で、元のnoisy実験の再現ではありません。
  IsingはKL目的と辺の並びを保持しますが、相互作用はNumPyで生成・固定し、同番号seedのTorch instanceとは異なります。
- [FMIRA](https://github.com/tucs7/FMIRA/tree/7d20bb6754aed252adb5343d0552a8b791c83e98)：
  3種類の生体fitness景観。URL、サイズ、SHA-256を `blackbox/sources.lock.json` に固定しています。
- [IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter)：`ioh==0.3.22`。
  [PBOの定義](https://iohprofiler.github.io/IOHproblem/PBO)。
- [LABSのFMA評価例](https://arxiv.org/abs/2507.21024)：energyとnegative merit factorは別IDで、結果を混ぜません。

性能比較ではbest-so-far、unique評価数、総実時間、fit・acquisition時間、複数のinstance・optimizer seedを
記録し、gapは最適値が検証済みの場合だけ報告してください。smoke testだけで手法を順位付けしないでください。
