# PyTorchLightningHydra

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)と[Hydra](https://github.com/facebookresearch/hydra)を使ったプログラム例です．
MNISTを学習します．

## Dockerの利用

Dockerに関するシェルスクリプトが`docker/`フォルダにまとめられています．
拡張子が`.sh`のファイルを実行すれば簡単にDockerを利用できます．

- `./docker/build.sh`: DockerfileからDockerイメージを作ります． Dockerイメージの名前は`PyTorchLightningHydra`です．
- `./docker/run.sh`: Dockerイメージ`PyTorchLightningHydra`のコンテナを作ります． コンテナの名前は`pytorch_lightning_hydra`です．このシェルスクリプトを実行したディレクトリとパス`$DATASET`が示すディレクトリがそれぞれコンテナ内の`/workspace/`と`/dataset/`に**マウントされ，完全に同期されます**．また，5000,6006,8888のポートをホストに割り当てます．環境変数は`.env`ファイルから読み込みます．
  -  引数: コンテナで常時実行するコマンド．デフォルトはfish
- `./docker/attach.sh`: `run`で実行したコマンドに復帰します．このコマンドが終了するとコンテナも終了します．
- `./docker/exec.sh`: 動作中のコンテナで新しいコマンドを実行します．
  -  引数: 実行するコマンド．デフォルトはfish
- `./docker/tensorboard.sh`: コンテナ内でtensorboardを実行します．
  -  引数: サーバーのポート．デフォルトは6006
- `./docker/mlflow.sh`: コンテナ内でmlflowを実行します．
  -  引数: サーバーのポート．デフォルトは5000
- `./docker/jupyter.sh`: コンテナ内でjupyter-labを実行します．**非常にセキュアでない**ので実行には注意して下さい．
  -  引数: サーバーのポート．デフォルトは8888

`.env`の作成には`.env.default`を参考にして下さい．

## PyTorch Lightningによる学習

学習のスクリプトは全て`src/`にまとめられています．
学習するには`src/train`を実行します．

ニューラルネットワークのモデルは`src/model`にまとめられています．

### ハイパーパラメータの変更

`src/train`は引数を受け付けます．
バッチサイズや学習率，使用するオプティマイザやモデル構造などを指定可能です．
次のコマンドを実行すると，バッチサイズを64，オプティマイザそSGDとして学習します．
```
python src/train.py hparams.batch_size=64 hparams.optimizer=SGD
```
Hydraのmultirun機能を利用して，複数のパラメータを網羅的にグリッドサーチすることもできます．
次のコマンドのように，`-m`オプションを加え，パラメータをcommaで区切ります．
```
python src/train.py hparams.batch_size=16,64,256 hparams.optimizer=SGD,Adam -m
```
詳細は[Hydraの公式サイト](https://hydra.cc/)を参照して下さい．

引数は全て`config/`にまとめられています．
`config/config.yaml`が大元のファイルです．
その他のyamlファイルはこのファイルから読み込まれます．

### 学習結果

学習結果は`outputs/`に保存されます．
Hydraのmultirunを使った場合は`multirun/`に保存されます．

保存されるのは学習したモデルの重み，Hydraの引数，log出力などです．

### ロガー

PyTorch Lightningのlogging機能により自動的にTensorBoardのログが保存されます．
ホスト側で`docker/tensorboard.sh`を実行し，指定したポートにブラウザからアクセスするとログが見れます．
Pytorch Lightningでは他にもMLflowやNeptuneなど多くのロガーが利用可能です．
詳細は[PyTorch Lightningの公式ドキュメント](https://pytorch-lightning.readthedocs.io/en/stable/loggers.html)を参照して下さい．

[MLflow](https://mlflow.org/)はローカルサーバーを立ててログを閲覧できるTensorBoardのようなロガーです．
MLflowのログは`mlruns`に保存されます．
ホスト側で`./docker/mlflow.sh`を実行し，指定したポートにブラウザからアクセスするとログが見れます．

[Neptune](https://ui.neptune.ai/)はオンラインにログデータを送信して，どこでもブラウザからオンラインで閲覧可能なロガーです．
Neptuneを利用するには，`config/loggers/`で使うyamlファイルにNeptuneLoggerを追加し，オンラインで入手したAPIキーを環境変数`NEPTUNE_API_TOKEN`に設定します．環境変数は`.env`ファイルで設定しましょう．**APIキーは他人に教えないように気をつけましょう**

## HydraのTab completion機能

[Hydra](https://github.com/facebookresearch/hydra)にはタブ補完機能が備わっています．
コマンドラインで`python hoge.py `まで入力しTabキーを入力すれば，`hoge.py`が入力として受け付ける候補のキーワードを補完できます．

この機能を有効化するシェルスクリプトが`hydra/`にまとめられています．
bashでは`source hydra/tab_completion_bash hoge.py`，fishでは`source hydra/tab_completion_fish hoge.py`とコマンドラインに入力すると有効化されます．
ここでhoge.pyは対象のpyファイルに置き換えて下さい．
ただし，
これが有効化される範囲はそのシェル内のみなので，シェルを起動するたびに実行する必要があります．
