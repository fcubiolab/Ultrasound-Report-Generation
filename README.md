# 超音波報告生成
本程式碼為「Ultrasound Report Generation with Cross-Modality Feature Alignment via Unsupervised Guidance」之實作。
我們提出了一個新穎的自動超音波報告生成框架，結合無監督與有監督學習方法以輔助報告生成過程。該框架利用無監督學習方法從超音波文本報告中提取潛在知識，作為先驗資訊來引導模型對齊視覺與文本特徵，從而解決特徵不一致的問題。此外，我們設計了一種全域語意比較機制，以提升生成更全面且準確醫學報告的效能。
![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/63fe3ae3-293a-45b1-af9a-099468c644fc)

## 主要結果
![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/c216ef5e-8bea-4ca5-8214-de339a136861)

## 實作說明
### 設定
- 在 `KMVE_RG/config.py` 先定義 `config.base_path`，其餘路徑皆由此組合而成。

### 執行分群
- 執行 ./knowledge_Distiller/knowledge_distiller.py 以獲取分群標籤。

### 執行訓練流程
- 執行 ./KMVE_RG/my_main.py 以訓練 SGF。

## 資料集
超音波資料集可於 [https://drive.google.com/file/d/11Aw3_ETNBtfT1W7eWifbsaexFqSsM5BB/view?usp=drive_link](https://drive.google.com/file/d/1-Fz9J58ntoO8ZoAEKzm3fxm4jvOWrAuz/view?usp=drive_link) 下載。
為評估本框架於不同類型超音波資料集上的表現，我們收集了乳腺、甲狀腺與肝臟三種資料集。具體來說，乳腺資料集包含 3521 位病患，甲狀腺資料集包含 2474 位病患，肝臟資料集包含 1395 位病患。

![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/d3bb3c79-7ad9-4cfa-92be-07a63734b4da)

## 系統運作說明

本專案是「Ultrasound Report Generation with Cross-Modality Feature Alignment via Unsupervised Guidance」的實作，主要目的是自動生成超音波醫學報告。以下簡要說明其運作流程：

### 1. 系統架構與流程
本系統結合了無監督與有監督學習方法，利用無監督學習從超音波報告中萃取潛在知識，並將這些知識作為先驗資訊，協助模型對齊視覺（影像）與文本（報告）特徵，解決特徵不一致的問題。系統還設計了全域語意比較機制，提升報告生成的完整性與準確性。

### 2. 主要步驟
1. **設定參數**  
   請先在 `KMVE_RG/config.py` 設定資料路徑與超參數。
2. **分群（知識蒸餾）**  
   執行 `Knowledge_Distiller/knowledge_distiller.py` 進行文本分群，獲取每筆資料的分群標籤，這些標籤會作為後續模型訓練的輔助資訊。
3. **模型訓練**  
   執行 `KMVE_RG/my_main.py` 開始訓練報告生成模型。
   - 影像經由視覺特徵擷取器（如 ViT）轉成特徵向量。
   - 報告文本經分詞與編碼，轉成詞向量。
   - 透過 Encoder-Decoder 架構（如 Transformer）將影像特徵與文本特徵對齊，並生成報告。
   - 訓練過程同時考慮報告生成損失、分群分類損失，以及語意相似度損失。
4. **評估與測試**  
   系統會自動計算 BLEU、METEOR、ROUGE、CIDEr 等指標，評估生成報告的品質。

### 3. 資料集
本專案支援乳腺、甲狀腺、肝臟三種超音波資料集，下載連結與數量詳見下方說明。

### 4. 主要檔案說明
- `KMVE_RG/config.py`：參數與路徑設定。
- `Knowledge_Distiller/knowledge_distiller.py`：文本分群與知識蒸餾。
- `KMVE_RG/my_main.py`：訓練主程式。
- `KMVE_RG/models/SGF_model.py`：模型架構。
- `KMVE_RG/modules/`：包含資料處理、特徵擷取、訓練、評估等模組。