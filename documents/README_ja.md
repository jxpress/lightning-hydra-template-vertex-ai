# Lightning Hydra Template Vertex AI

学習フレームワークである [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)とハイパーパラメーター管理の [Hydra](https://github.com/facebookresearch/hydra)を用いることで、数行のコード変更で並行で学習出来る等、様々な恩恵があり、効率的に学習を進められます。この2つのパッケージをもとに作成された素晴らしい[学習テンプレートコード](https://github.com/ashleve/lightning-hydra-template)が公開されています。PyTorch Lightning やHydraの詳細はテンプレートコードの[README](/documents/README.original.md)を御覧ください。

このレポジトリは、このテンプレートコードが[Vertex AI](https://cloud.google.com/vertex-ai) で実行できるようにコードを追加したものになります。
![main_theme](/documents/images/main_readme.png)
<br>

## 💡　一般公開する理由


Vertex AIはGoogle Cloud Platformの統合機械学習プラットフォームであり、Vertex AIを利用することで以下のものが簡単に実行できます。 ( (★)はこのレポジトリで実装可能、(※)が実行可能なサンプルコードも公開予定)
- 学習時だけ、GPUが起動する学習 (★)
- 並列学習でハイパーパラメータの最適化 (★)
- データの前処理、学習、評価、デプロイを切り分け、それぞれをpipelineでつなげ学習する学習 (※)
- 定期的、または、トリガーによるGPUマシーンでの学習


しかし、Vertex AIとHydraではコマンドライン引数の渡し方が異なるため、相性が悪く、Hydraで書かれたコードをVertex AIで実行するためには工夫が必要になります。
本レポジトリではその工夫がなされており、苦労なくVertex AIで学習できるようにしました。

詳しい問題点や解決法は[こちらのブログ](https://tech.jxpress.net/entry/2022/05/13/113011)を御覧ください。

<br>

## 🚀  Vertex AIで学習するための方法
### step 1. テンプレートコードを編集し独自の学習コードを作成する。 (オプショナル)
自分たち独自のAIを作成したい場合は、テンプレートコードを編集し、学習が完了することを確かめてください。
Vertex AIでの動作確認だけされたい場合は、テンプレートコードをそのまま実行すると、MNISTの分類の学習が行われます。

<br>

### step 2. Dockerで学習が出来ることを確認する
Vertex AIではDocker Imageを用いて学習が行われるため、Dockerでの動作確認が必要になります。
その際、
```bash
make train-in-docker
```
をroot ディレクトリで実行して動作確認をしてください。
GPUでの動作確認等のOptionは[docker-compose.yaml](/docker-compose.yaml)で調整ができます。

<br>


### step 3. GCPのアカウントを用意する
GCPのアカウントがない場合は[こちら](https://cloud.google.com/docs/get-started)から、GCPのアカウントを用意してください。
本レポジトリでは、[Vertex AI](https://cloud.google.com/vertex-ai/docs/start)と[Artifact Registry](https://cloud.google.com/artifact-registry)を利用するので、GCPのそれぞれのAPIを有効化してください。

次に、Artifact RegistryにDocker imageをpushするために[dockerのレポジトリを作成](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#overview)する。

その後、[Imageの名前を決定する](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)。

<br>


### step 4-1. Custom jobを実行したい場合
- step 3で設定した、Artifact Registryに設定した、Imageの名前とタグを[vertex_ai/configs/custom_job/dafault.yaml](/vertex_ai/configs/custom_job/dafault.yaml)のimageUriに設定する。
- [vertex_ai/scripts/custom_job/create_job.sh](/vertex_ai/scripts/custom_job/create_job.sh)のregion, gcp_projectを設定する
- ルートフォルダで
```bash
make create-custom-job
```
を入力することで、docker のbuildとpushが行われ、pushされたイメージを用いてVertex AIのCustom jobが開始される。

学習状況はGCPのVertex AIのトレーニングの[CUSTOM JOBS](https://console.cloud.google.com/vertex-ai/training/custom-jobs)で確認が出来る


<br>


### step 4-2. Hyperparameter tuning job
- step 3で設定した、Artifact Registryに設定した、Imageの名前とタグを[vertex_ai/configs/hparams_tuning/default.yaml](/vertex_ai/configs/hparams_tuning/default.yaml)のimageUriに設定する。
- ハイパラ調整で最適化したいメトリックを[configs/hparams_search/vertex_ai.yaml](/configs/hparams_search/vertex_ai.yaml)で設定する。
- ハイパラ調整で最適化したいパラメーターを[vertex_ai/configs/hparams_tuning/default.yaml](/vertex_ai/configs/hparams_tuning/default.yaml)のstudySpecで設定する
- [vertex_ai/scripts/hparams_tuning/create_job.sh](/vertex_ai/scripts/hparams_tuning/create_job.sh)のregion, gcp_projectを設定する
- ルートフォルダで
```bash
make create-hparams-tuning-job
```
を入力することで、docker のbuildとpushが行われ、pushされたイメージを用いてVertex AIのCustom jobが開始される。

学習状況はGCPのVertex AIのトレーニングの[HYPERPARAMETER TUNING JOBS](https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs)で確認が出来る

<br>



# 🔧　変更点
本レポジトリでは、[学習テンプレートコード](https://github.com/ashleve/lightning-hydra-template)から以下の変更をしました。
- configs/hparams_search/vertex_ai.yaml
    - Vertex AIのHyperparameter Tuningで利用
- Makefile 
    - dockerとVertex AIに関わるコードの追加
- Vertex AI専用のフォルダとコード
    - configs
        - 設定関連のyamlの追加
    - script
        - Vertex AIで実行するために必要なコードの追加
- requirements.txt
    - Vertex AIで必要なライブラリの追加
- README.md
    - 新たなREADMEの追加. もともとのテンプレートコードのREADMEはDocumentsのフォルダに移動
- Documents
    - READMEの日本語バージョン
    - 翻訳したブログ
        - hydraとVertex AIについて、詳細に書かれたブログを英訳


<h1 id="appendix"> 📝 補足</h1>

[JX通信社](https://jxpress.net/)では、チームでの開発力や開発速度を高めるために、学習テンプレートコードを作成し運用しています。
本レポジトリは、JX通信社で利用している学習テンプレートコードから、Vertex AIで学習するためのcodeだけを [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)に移植しました。
JX通信社のテンプレートコードについて、詳しくは[属人化しがちなR&Dをチーム開発するためのJX通信社での工夫](https://tech.jxpress.net/entry/2021/10/27/160154)や[ヘビーユーザーが解説するPyTorch Lightning](https://tech.jxpress.net/entry/2021/11/17/112214)のブログを御覧ください。

# 😍主なコントリビューター
本レポジトリへの移植は[Yongtae](https://github.com/Yongtae723)が行いましたが、開発は[Yongtae](https://github.com/Yongtae723)が発案・提案を、[near129](https://github.com/near129)がコード開発を主導しました。

# 改善したいポイント
- .shでconfig fileから値を直接値を取得しているが、綺麗な実装ではない 