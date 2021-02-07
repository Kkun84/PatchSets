# PatchSets

パッチ集合を入力としてクラス分類をするニューラルネットワークの研究．
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)と[Hydra](https://github.com/facebookresearch/hydra)を使っています．

## Dockerの利用

Dockerに関するシェルスクリプトが`docker/`フォルダにまとめられています．
拡張子が`.sh`のファイルを実行すれば簡単にDockerを利用できます．

| ファイル | 説明 | 引数 (デフォルト値) |
| --- | --- | --- |
| `./docker/build.sh` | DockerfileからDockerイメージを作ります． Dockerイメージの名前は`PatchSets`です． |
| `./docker/run.sh` | Dockerイメージ`PatchSets`のコンテナを作ります． コンテナの名前は`patch_sets`です．このシェルスクリプトを実行したディレクトリとパス`$DATASET`が示すディレクトリがそれぞれコンテナ内の`/workspace/`と`/dataset/`に**マウントされ，完全に同期されます**．また，5000,6006,8888のポートをホストに割り当てます．環境変数は`.env`ファイルから読み込みます． | コンテナで常時実行するコマンド (fish) |
| `./docker/attach.sh`| `run`で実行したコマンドに復帰します．このコマンドが終了するとコンテナも終了します． |
| `./docker/exec.sh` | 動作中のコンテナで新しいコマンドを実行します． | 実行するコマンド (fish) |
| `./docker/tensorboard.sh` | コンテナ内でtensorboardを実行します． | サーバーのポート (6006) |
| `./docker/mlflow.sh` | コンテナ内でmlflowを実行します． | サーバーのポート (5000) |
| `./docker/jupyter.sh` | コンテナ内でjupyter-labを実行します．**非常にセキュアでない**ので実行には注意して下さい． | サーバーのポート (8888) |

`.env`の作成には`.env.default`を参考にして下さい．

## PyTorch Lightningによる学習

学習のスクリプトは全て`src/`にまとめられています．
学習するには`src/train`を実行します．

ニューラルネットワークのモデルは`src/model.py`に書かれています．
`src/set_module/`に集合を入力とするモデルを簡単に書くためのモジュールが書かれています．

### ハイパーパラメータの変更

`src/train`は引数を受け付けます．
バッチサイズや学習率，使用するオプティマイザやモデル構造などを指定可能です．
次のコマンドを実行すると，バッチサイズを64，オプティマイザをSGDとして学習します．
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

## Visual Studio Codeの利用

使用するエディタにはVisual Studio Code(以下vscode)をおすすめします．
ダウンロードは[こちら](https://code.visualstudio.com/)です．
vscodeでは，コンテナへのアタッチや高度なデバッグなど，豊富な機能が利用できます．

vscodeの設定ファイルは`.vscode/`にまとめられています．

### 拡張機能

おすすめの拡張機能が`.vscode/extensions.json`にまとめられています．
vscodeの**フォルダーを開く**でこのプロジェクトを開けば，まとめてインストールすることもできます．

### コンテナへのアタッチ

vscodeの作業を動作中のコンテナ内部で行うことができます．
これは，vscode上で**Remote - Containers拡張機能**をインストールすることで利用できます．
拡張機能は[こちら](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)です．

実際にコンテナ内で作業するには，

1. 事前にコンテナを作る（ターミナルで`./docker/run.sh`を実行する）．
1. vscodeのコマンドパレットから**Remote-Containers: Attach to Running Container**を選択する．
1. 動作中のコンテナ一覧が表示されるので，対象のコンテナを選ぶ．

とします．
vscodeの作業をコンテナ内で行う方法は他にもあります．
これらの詳細を知るにはRemote - Containersの拡張機能について調べましょう．

### デバッグ

**F5キー**でデバッグができます．
コンテナにアタッチしていればコンテナ内でもデバッグが可能です．
詳細な説明はしませんが，非常に便利なので利用しましょう．

デバッグの設定は`.vscode/launch.json`に記述されています．
launch.jsonには以下の行が書かれています．
```
"args": ["debug=True", "experiment_name=__debug__"]
```
これは，デバッグ時の引数を指定しており，
```
python hoge.py debug=True experiment_name=__debug__
```
と実行されると考えればよいです．
デバッグ時に動作させたい内容に応じてここは変えましょう．

train.pyを`debug=True`として実行すると，Pytorch Lightnigの[fast_dev_run](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#fast-dev-run)が有効となります．
これはバグを探すための機能です．
これが有効だとtrain, valid, testがそれぞれバッチ数1で実行されます．

train.pyを`experiment_name=__debug__`として実行すると実験の名前が__debug__となります．
これはロギングへ影響するもので，実験結果を保存する名前やグループを設定しています．
デバッグのログは通常実行のログとは区別したいことがほとんどなので，このようにしています．
