本任務希望可以對山坡圖片做Instance Segmentation，要區分的類別有「堤防」、「天空」、「土石坍塌」、「Green」、「水」等。
## CV Task:Instance Segmentation
#### Object Detection
<img src="./figure/mrcnn01.png" width="400" alt="物件偵測"/>
物件偵測:用矩行框把物件一個一個框出來，可以區分個體，但不能切割出輪廓。

#### Semantic Segmentation
<img src="./figure/mrcnn02.png" width="400" alt="語意分割"/>
語意分割:可以精細的切出輪廓但不能區分個體，像是這張圖片可以知道這團都是人但不知道有幾個人

#### Instance Segmentation
<img src="./figure/mrcnn03.png" width="400" alt="語意分割"/>
Instance Segmentation 會對圖片中的每一個像素點做分類,並且區分不同的物件。圖中的人可以區分個體又可以切割出輪廓。
  
Source : https://www.muhendisbeyinler.net/mask-r-cnn-bir-nesne-tanima-algoritmasi/


## pre-processing : Label, Transform 
使用Labelme對圖片座標註，每張圖片都有一個對應的XML檔，要把XML檔轉成可以餵給模型的格式，使用`labelme_json_to_dataset.exe`來進行轉檔。  
```
python run_json.py
```

要注意的是，由於此任務是做Instance Segmentation，因此在標註、轉檔時要特別注意「希望區分個體的類別」。

錯誤示範













## Instance Segmentation
input:  
<img src="https://upload.cc/i1/2023/01/04/9PItVO.jpg" width="700" alt="input原圖"/>

output:  
<img src="https://upload.cc/i1/2023/01/04/NyS70l.png" width="700" alt="output預測圖片"/>


## Reference

