# funko-pop-cnn

#### Definition of a “Funko Pop!”: ​

“Small figurines renowned for their exaggerated features, […]. They depict popular characters from a wide range of leading franchises and genres” (https://toysforapound.com). ​

#### Goal: ​

To design an AI model to identify different attributes such as Hair, Items in hand, and Gender.​

#### Real-World Applications:​

Enhancing manufacturing efficiency by implementing AI-enabled defect detection systems on production lines​

![image](https://github.com/cmansoo/funko-pop-cnn/assets/70994968/560b883f-ff72-4424-9705-9d99768fee59)

#### Dataset Contains:​

8 different angles of 32 Funko Pops for a total of 256 Records (Images).​

Gender Classification most imbalanced and complex due to 3 classes.​

| Human/Non-Human​ | Gender​    | Facial Hair​ | Glasses​ | Hat​    | Item in Hand​ |
| ---------------- | ---------- | ------------ | -------- | ------- | ------------- |
| Human: 19​       | Male: 18​  | Yes: 6​      | Yes: 2​  | Yes: 7​ | Yes 17​       |
| Non-Human: 13​   | Female: 1​ | No: 26​      | No: 30​  | No: 25​ | No 15​        |
| ​                | Other: 13​ | ​            | ​        | ​       | ​             |


### MANSOONET- CNN FROM SCRATCH​

Preprocessing: Image Augmentation & train/validation/test​

Model structure: 4 Convolutional Layers -> Flatten -> 8 Dense Layers -> Output Layer (3 class labels)​

 Total ~47 million trainable parameters​

| Non-trainable parameters​ |
| ------------------------- |
| Input size​               | (224, 224, 3)​ |
| Filters​                  | 72 -> 144 -> 216 -> 360​ |
| Filter size​              | (11,11) -> (7,7) -> (5,5) -> (3,3)​ |
| Activation​               | ReLU​ |
| Pooling​                  | Max Pooling​

(3,3) & (2,2)​ |
| ​                         |

#### Full Architecture

<-- image goes here -->


