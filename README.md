Fruit Classification and Rotten Fruit Detection


เหตุผลที่เลือกหัวข้อนี้

ผมเลือกหัวข้อนี้เพราะว่า หลังจากได้ดูวิดีโอใน YouTube เกี่ยวกับการทำฟาร์มในประเทศเนเธอร์แลนด์ ซึ่งเป็นประเทศที่มีสภาพอากาศหนาวเย็นแต่สามารถผลิตผลไม้และพืชผลได้อย่างมีประสิทธิภาพมาก ในคลิปนั้นผมได้เห็นการนำ Deep Learning มาใช้ในการคัดแยกและตรวจสอบความสดของผลไม้ เพื่อให้สามารถเก็บเกี่ยวได้ในช่วงเวลาที่เหมาะสมที่สุด สิ่งนี้ทำให้ผมรู้สึกสนใจ และอยากลองพัฒนาโมเดลในลักษณะเดียวกันด้วยตนเอง แต่ว่าเนื่องจากไม่สามารถ Label ข้อมูลความสดและ ระดับของความสดของผลไม้ได้ด้วยตัวเองจึง เลือกหัวข้อ Classification and Rotten Detection แทน


ทำไมถึงต้องใช้ Deep Learning ในการแก้ปัญหา

การแยกประเภทผลไม้และการตรวจสอบความเน่า หรือ ตำหนิของผลไม้นั้นเป็นงานที่มีความซับซ้อน เนื่องจากผลไม้มีลักษณะที่หลากหลาย และ ตำหนิของผลไม้ สามารถอยู่ได้ทุกตำแหน่งของผลไม้ ดังนั้นปัญหานี้จึงมีความซับซ้อนที่จะต้องใช้ Deep Learning ในการแก้ปัญหา 

ลักษณะเด่นของ Deep Learning เลยคือ สามารถเรียนรู้ลักษณะของผลไม้จากภาพได้เลย โดยไม่ต้องกำหนดกฎ ขึ้นมาเอง เหมาะสมกับปัญหาที่มีความซับซ้อน

ข้อเสียของ Deep Learning คือ จำเป็นต้องใช้ข้อมูลจำนวนมาก และ ต้องมี GPU และ จำนวน VRAM ที่มีความสามารถมากพอ ที่จะใช้ในการ Train model นี้ได้ ตามขนาดความใหญ่และซับซ้อนของรูปภาพ


สถาปัตยกรรม

โมเดลที่ใช้เป็น Convolution Neural Network  หรือที่เรียกกันด้วยชื่อย่อว่า CNN ซึ่งเป็น Neural Network ที่มีลักษณะคล้าย feed forward network แต่ต่างกันตรงที่ CNN นั้นถูกออกแบบมาเพื่อให้ทำงานกำรูปโดยเฉพาะ เนื่องจากการทำ Convolution เพื่อที่จะหาจุดเด่นของ feature และทำการลดขนาดพื้นที่ด้วยการทำ Pooling เพื่อดึงลักษณะเฉพาะของ ข้อมูลที่ถูกใส่เข้าไป เพื่อแยกแยะ และ เรียนรู้แบบมีประสิทธิภาพมากกว่าการใช้ FFNN ทั่วไปในการวิเคราะห์รูปภาพ โดย Activation Function ที่เลือกมาคือ SiLU ซึ่งจะมีความคล้ายกับ ReLU แต่ว่าใหม่กว่า และ เหมาะที่จะใช้กับ BatchNorm มากกว่า อีกทั้งยังไม่ทำให้เกิดปัญหา dead neuraon เมือนกับ ฟังค์ชันของ ReLU อีกด้วย

<img width="2748" height="3981" alt="image" src="https://github.com/user-attachments/assets/93ecf8dc-919e-4778-a5c6-164fe73a61e0" />

อธิบายโค้ดที่ใช้สร้าง Model

โค้ดในส่วนของการ โหลดข้อมูลจาก kaggle แล้วนำมาใช้:

<img width="596" height="148" alt="image" src="https://github.com/user-attachments/assets/b61414cb-c89c-4965-a714-2782d0e70cd7" />

โค้ดในการนำข้ำมูลที่โหลด มา ใส่ใน format ที่ถูกต้องเพื่อให้โมเดลสามารถใช้ในการเทรนได้:

<img width="717" height="361" alt="image" src="https://github.com/user-attachments/assets/be8e164a-5c3e-4dff-b18f-4b72faceff78" />

โดยใน dataset ของ kaggle ที่ได้ไปหามานั้น พบว่า มีส่วนของการ train และ test แต่เนื่องจากลอง train และ evaluate ด้วย test นั้นรู้สึกว่าข้อมูลของการ test ที่ให้มานั้นไม่ได้ดีนัก จึงทำการ split ข้อมูล test ออกมาจากการ train แทนโดยใช้อัตราส่วนที่ 80% Train และ 20% Test

โค้ดส่วน  convolution ของ Model นี้:

<img width="558" height="262" alt="image" src="https://github.com/user-attachments/assets/d9694bf0-5439-4150-b850-f64d73ea432c" />

โดยในส่วนของการสร้าง ConvBlock นั้น ได้มีการออกแบบให้เป็นโครงสร้างพื้นฐานของชั้นคอนโวลูชันในโมเดล เพื่อให้สามารถเรียกใช้งานซ้ำได้หลายครั้ง โดยภายในคลาสจะมีการประกอบด้วยชั้น Convolution2D, Batch Normalization, และ Activation function SiLU พร้อมด้วย Dropout2D เพื่อช่วยลดการเกิด overfitting ส่วน forward เป็นตัวกำหนด path ของข้อมูลที่ถูกส่งไปในแต่ละ layer ของ ConvBlock 

โค้ดของ Model:

<img width="482" height="651" alt="image" src="https://github.com/user-attachments/assets/007f596d-8afd-44f9-919f-9160a4483c2e" />

โดยโค้ดของการสร้างโมเดล มีทั้งหมด 5 layers ด้วยกันโดย Layer แรกสุดเป็น Layer ที่ชื่อว่า stem มีเอาไว้ เพื่อรับภาพสีขนาด 128x128x3 คือ 128 pixels x 128 pixels x 3 RGB, Red Green และ Blue และแปลงเพื่อให้ Model สามารถเข้าใจได้ง่ายขึ้น ต่อมาเป็น Layers 1 ถึง 3 Layers เหล่านี้จะค่อยๆ แปลงจาก feature ที่มีความง่ายไปเป็น feature ที่มีความ complex มากยิ่งขึ้น และในแต่ละ Layers เหล่านี้จะมี MaxPool2D ซึ่งทำหน้าที่ย่อขนาดลงครึ่งหนึ่ง เพื่อทำให้โมเดลสามารถเรียนรู้ feature จากง่ายไปซับซ้อนได้ และ ลดความ noisy ของภาพ

หลังจากนั้นก็จะออกมาสู้ Layer สุดท้ายคือ head ซึ่งเป็น layer ที่เอาไว้ Classified หรือเป็น Layer ของ output ซึ่ง Layer นี้ประกอบด้วย AdaptiveAvgPool2D ซึ่งเอาไว้ลดขนาดจาก Layers ที่ 3 ให้เหลือ 1x1 แล้วจึงนำมา Flatten เพื่อแปลงเป็น Vector ยาว 256 จากนั้นผ่าน neuron แบบ Linear 256 ซึ่งมี SiLU เป็น Activation Function และ ทำการ DroupOut เพื่อทำให้ model นั้น Generalize ได้ดี ก่อนที่จะเข้า Linear ที่ย่อ 256 ไป 6 ซึ่งเป็นจำนวนของ Class ที่มีทั้งหมดของข้อมูลที่ที่ต้องการจะวิเคราะห์ด้วย โมเดลนี้

Config ของ Model, Optimizer, Scheduler, Lossfunction และ Scaler:

<img width="600" height="267" alt="image" src="https://github.com/user-attachments/assets/6ba3cc40-f4a8-4795-b8d7-6e5828e371be" />

Loss function ใช้ CrossEntropyLoss, Optimizer ใช้ AdamW, Scheduler ใช้ CosneAnnealingLR เป็นการทำให้ Learning Rate ลดขนาดลง เพื่อให้ในช่วงแรกๆ สามารถ train โมเดลได้อย่างรวดเร็ว หลังจากนั้นที่โมเดลได้เข้าที่เข้าทาง จึงทำการลด Learning Rate เพื่อให้โมเดลปรับ Accuracy ของตัวเองให้ดีขึ้นช้าๆ และ เสถียร เพราะว่าช่วงแรกของการ Train นั้นเราสามารถ ไม่ต้องกังวลผลของ Learning Rate ที่มีขนาดมากเกินไป เนื่องจากว่า จากที่ผมได้พบมา การที่ Accuracy สั่นที่เป็นผลกระทบของ Learning Rate จะเกิดตอนที่ Accuracy ของ Model สูงเป็นส่วนใหญ่ ดังนั้นจะเป็นเหมือนการ Fine Tuning Hyperparameter ด้วยตัวเอง

Evaluate Function:

<img width="546" height="237" alt="image" src="https://github.com/user-attachments/assets/3d1d4386-2458-4494-8db7-03f23e796858" />

โค้ดของการ Train:

<img width="504" height="575" alt="image" src="https://github.com/user-attachments/assets/57a943fb-243c-4d11-bff1-1b5487375c87" />

ส่วนของการ Plot:

<img width="632" height="376" alt="image" src="https://github.com/user-attachments/assets/f7304fee-b667-49e8-94e5-c6500b8f0516" />

อธิบายวิธีในการ train

การตั้งค่าของการ train: 
	Epochs = 15
	Learning Rate = 5e-3
	Weight Decay = 1e-4
	grad_clip = 1.0 ป้องกัน Gradient ระเบิดคือจะ Clip ที่ 1
	use_amp = True

Loss Function ที่ใช้คือ CrossEntropyLoss เพราะว่าใช้สำหรับจำแนกหลาย Class
Optimizer adamw ใช้ weight_decay เพื่อลด overfitting บนพารามิเตอร์
Scheduler: ลด Learning Rate ทุก epochs เป็นเหมือนการ finetune ด้วยตัวเอง
Automatic Mixed Precision หรือ Amp ใช้เพื่อลดการใช้ Computation power
Device handling ใช้ cuda เมื่อมีให้ใช้ เพื่อความเร็วในการ train

การ train จะเริ่มจากการโหลดข้อมูลเข้า xb และ yb
Clear gradient ด้วย optimizer.zero_grad(set_to_none=True)
logits = model(xb); loss = criterion(logits, yb) เพื่อคำนวน forward
จากนั้นคำนวน back propagation ด้วย scaler.scale(loss).backward()
ถ้ามี grad_clip จะยกเลิก clip_grad_norm_ เพื่อจำกัด norm ของ parameter
Update parameter ด้วย scaler.step(optimizer); scaler.update()
จากนั้นคำนวน tran loss, train accuracy, test loss, test accuracy แล้วเก็บเอาไว้
จากนั้นบันทึกลง History
เก็บ model ที่ Accuracy มากที่สุดเผื่อ model overfit จะได้ย้อนกลับมาใช้ model ในตอนที่ accuracy ดีที่สุด
ปรับ learning rate ต่อ epoch ด้วย scheduler.step()
หลังจบทั้งหมด แสดงเวลารวม และ โหลดน้ำหนักที่ดีที่สุด กลับสู่โมเดลหากมี best_state

Dataset ที่ใช้

Dataset ที่ใช้เป็น dataset ชนิดภาพ นำมาจาก Kaggle Download ผ่าน Colab ด้วย kaggle api จาก Library ของ Kaggle ซึ่งใน Dataset จะมี Train และ Set ที่เตรียมมาให้ แต่เนื่องด้วย Test folder นั้นมีปัญหา จึงทำการ สร้าง Test ด้วยการ Split ข้อมูลจาก Folder Train ออกมา, รูปภาพในใน Dataset นั้นเป็นรูปภาพของผลไม้ที่เป็น RGB และ มีจำนวนของ Pixels ไม่ทำกัน เลยทำการย่อให้มีขนาด 128x128 ทุกภาพเท่ากัน ใน Dataset นั้นมีทั้งหมด 6 Class ซึ่งประกอบด้วย กล้วย ส้ม และ แอปเปิ้ล ซึ่งทั้ง 3 ผลไม้เหล่านี้สามารถแบ่งออกได้อีก เป็นผลไม้สด และผลไม้เน่า จึงรวมกันทั้งหมด 6 class





อธิบายวิธีในการ ประเมิณ

หลังจากทำการเทรนโมเดล TinyCNN จนครบ 15 epoch แล้ว ได้ทำการประเมินผลด้วยTest set ที่แบ่งไว้ในอัตรา 20% ของข้อมูลทั้งหมด เพื่อวัดประสิทธิภาพของโมเดลในการจำแนกภาพผลไม้สดและผลไม้เสีย โดยใช้ตัวชี้วัด ที่เหมาะสมกับงานจำแนกหลายคลาส ได้แก่ Loss, Accuracy, Precision, Recall และ F1-score

<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/91c4139f-c7a2-4a73-8623-9ff2eafb010a" />

จากกราฟ แสดงให้เห็นว่า Loss ของทั้งชุด Train และ Test ลดลงอย่างต่อเนื่องตลอดช่วงการฝึก ซึ่งสามารถบอกได้ว่าโมเดลมีการเรียนรู้และปรับพารามิเตอร์ได้อย่างเสถียร โดยค่า Test loss ไม่สูงกว่าชุด Train และ แสดงถึงการเกิด overfitting เพียงเล็กน้อยหรือแทบไม่มีเลย

<img width="589" height="590" alt="image" src="https://github.com/user-attachments/assets/dc992f42-d862-4651-8d5d-baba2b24d89a" />

ภาพ Confusion Matrix แสดงให้เห็นการจำแนกผลลัพธ์ของแต่ละคลาส
“freshapple” และ “rottenbanana” ถูกจำแนกได้ถูกต้องเกือบทั้งหมด
มีการสับสนระหว่าง “rottenapple” และ “rottenorange” เพียงเล็กน้อย
Class ที่เหลือจำแนกได้ถูกต้องอยู่ในเกณฑ์ที่น่าพอใจ


ซึ่งสะท้อนว่าโมเดลสามารถแยกความแตกต่างระหว่างภาพสดและภาพเน่าได้อย่างมีประสิทธิภาพ

<img width="640" height="557" alt="image" src="https://github.com/user-attachments/assets/77f4e251-0d8e-4be1-835b-73b3ca12e1aa" />

จากตารางผลลัพธ์การคำนวณ พบว่าโมเดลมี
Macro Precision = 0.97, Macro Recall = 0.97
ละ Accuracy รวม = 97%
ค่าดังกล่าวแสดงว่าโมเดลมีความแม่นยำสูงและสามารถตรวจจับตัวอย่างของแต่ละคลาสได้ดีโดยไม่มีการลำเอียงต่อคลาสใดคลาสหนึ่ง โดยเฉพาะคลาส rottenbanana และ freshbanana ที่มีค่า Recall และ Precision เกือบ 1.00



อ้างอิงและงานที่เกี่ยวข้อง

Dataset ที่ใช้:
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

 Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).

Dataset: “Fruits Fresh and Rotten for Classification,” Kaggle (sriramr, 2021)
