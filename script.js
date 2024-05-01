function startQuiz() {
  document.querySelector('.homepage-container').style.display = 'none';
  document.querySelector('.quiz-container').style.display = 'block';
}
const quizData = [
  {
    Question: "You are part of a Machine Learning team that has several large CSV datasets in Amazon S3. The team have been using these files to build a model in Amazon SageMaker using the Linear Learner algorithm. The time it takes to train these models is taking hours to complete. The team’s leaders need to accelerate the training process. What can a Machine Learning Specialist do to address this concern?",
    OptionA: "Use Amazon Kinesis to stream the data into Amazon SageMaker.",
    OptionB: "Create SageMaker Hyperparameter auto tuning job using ml.m5 instance types to find optimized hyperparameter.",
    OptionC: "Use Amazon SageMaker's Pipe input mode.",
    OptionD: "Use AWS Glue to transform the data from the CSV format into JSON.",
    CorrectAnswer: "Use Amazon SageMaker's Pipe input mode.",
    Description: "With Pipe input mode, the data is streamed directly to the algorithm container while model training is in progress. This is unlike File mode, which downloads data to the local Amazon Elastic Block Store (EBS) volume prior to starting the training. Using Pipe mode your training jobs start faster, use significantly less disk space and finish sooner. This reduces your overall cost to train machine learning models. Linear Learner Algorithm - Amazon SageMaker Linear Learner Algorithm - Amazon SageMaker"
  },
  {
    Question: "A machine learning model is being created using Amazon's Factorization Machines algorithm to help make click predictions and item recommendations for new customers. Which of the following would be candidates during the training process?",
    OptionA: "Making inferences to the model in application/csv format.",
    OptionB: "Creating a regression model where the testing dataset is scored using Root Mean Square Error (RMSE).",
    OptionC: "Using sparse data in CSV format as training data.",
    OptionD: "Creating a binary classification model where the testing dataset is scored using Binary Cross Entropy (Log Loss), Accuracy, and F1 Score.",
       CorrectAnswer: "Creating a regression model where the testing dataset is scored using Root Mean Square Error (RMSE).",
   
    Description: "The factorization machine algorithm can be run in either in binary classification mode or regression mode. In regression mode, the testing dataset is scored using Root Mean Square Error (RMSE). In binary classification mode, the test dataset is scored using Binary Cross Entropy (Log Loss), Accuracy (at threshold=0.5) and F1 Score (at threshold =0.5). For training, the factorization machines algorithm currently supports only the recordIO-protobuf format with Float32 tensors. CSV format is not a good candidate. For inference, factorization machines support the application/json and x-recordio-protobuf formats. Factorization Machines Algorithm - Amazon SageMaker"
  },
  {
    Question: "You are working for a company with strict compliance and data security requirements that requires that data is encrypted at all times, including at rest and in transit within the AWS cloud. You have been tasked with setting up a streaming data pipeline to move their data into the AWS cloud. What combination of tools allows this to be implemented with minimum amount of custom code?",
    OptionA: "Use SHA-256 custom managed keys to encrypt data before using the PutRecord or PutRecords API call. Use Kinesis Data Analytics to transform and decrypt data before using Kinesis Firehose to output results into Amazon S3",
    OptionB: "Kinesis only supports encryption of data once it is loaded into AWS cloud. Use Kinesis Data Analytics and a KMS key to transform and decrypt data before using Kinesis Firehose to output results into Amazon S3",
    OptionC: "Encrypt data with Amazon Kinesis Consumer Library (KCL), decrypt data with the Amazon Kinesis Producer Library (KPL), and use AWS KMS to manage keys",
    OptionD: "Encrypt data with the Amazon Kinesis Producer Library (KPL), decrypt data with Amazon Kinesis Consumer Library (KCL), and use AWS KMS to manage keys",
    CorrectAnswer: "Encrypt data with the Amazon Kinesis Producer Library (KPL), decrypt data with Amazon Kinesis Consumer Library (KCL), and use AWS KMS to manage keys",
      Description: "Encrypt data with the Amazon Kinesis Producer Library (KPL), decrypt data with Amazon Kinesis Consumer Library (KCL), and use AWS KMS to manage keys"
  },
  {
    Question: "You are a machine learning specialist building a model to determine the location (latitude and longitude) from different images taken and posted on a social media site. You've been provided with millions of images to use for training stored in Amazon S3. You've written a Java script to read the images from Amazon S3, extract pixels, and convert latitude and longitude data into CSV format to train the model with. Which service is the best candidate to distribute the workload and create the training dataset?",
    OptionA: "AWS Glue",
    OptionB: "Amazon EMR",
    OptionC: "Amazon Athena",
    OptionD: "SageMaker GroundTruth",
    CorrectAnswer: "Amazon EMR",
       Description: "Both Amazon EMR and AWS Glue provide ETL capabilities, but Amazon EMR is generally faster, less costly, and has more options than AWS Glue. Reference: What Is Amazon EMR? - Amazon EMR"
  },
  {
    Question: "You are in charge of training a deep learning (DL) model at scale using massively large datasets. These datasets are too large to load into memory on your Notebook instances. What are some best practices to use to solve this problem and still have fast training times?",
    OptionA: "Pack the data in parallel, distributed across multiple machines and split the data into a small number of files with a uniform number of partitions.",
    OptionB: "Once the data is loaded onto the instances, split the data into a small number of files and partitioned, then the preparation job can be parallelized and thus run faster.",
    OptionC: "Once the data is split into a small number of files and partitioned, the preparation job can be parallelized and thus run faster.",
    OptionD: "Use a fleet of RAM intensive ml.m5 EC2 instances with MapReduce and Hadoop installed onto them. Load the data in parallel to the cluster to distribute across multiple machines.",
       CorrectAnswer: "Pack the data in parallel, distributed across multiple machines and split the data into a small number of files with a uniform number of partitions.",
   
    Description: "When you perform deep learning (DL) at scale, for example, datasets are commonly too large to fit into memory and therefore require pre-processing steps to partition the datasets. In general, a best practice is to pack the data in parallel, distributed across multiple machines. You should do this in a single run, and split the data into a small number of files with a uniform number of partitions. When the data is partitioned, it is readily accessible and easily fed in as batches across multiple machines. When the data is split into a small number of files, the preparation job can be parallelized and thus run faster. You can do all of this using frameworks such as MapReduce and Apache Spark. Running an Apache Spark cluster on Amazon EMR provides a managed framework that can process massive quantities of data. Power Machine Learning at Scale"
  },
  {
    Question: "How can a Jupyter Notebook instance read data from an S3 bucket encrypted with SSE-KMS?",
    OptionA: "Import an external key into KMS and use it to securely read the data.",
    OptionB: "Encrypted data in S3 cannot be accessed through Notebook instances.",
    OptionC: "Ensure the Notebook instance role is an administrator who can administer the KMS key.",
    OptionD: "The Jupyter Notebook instance's associated IAM role must have a policy granting 'kms:Decrypt' permission for the specific KMS key used to encrypt the S3 bucket.",
    CorrectAnswer: "The Jupyter Notebook instance's associated IAM role must have a policy granting 'kms:Decrypt' permission for the specific KMS key used to encrypt the S3 bucket.",
      Description: "When data is stored in an S3 bucket and encrypted using SSE-KMS, the data is encrypted with a KMS key. To read this data, a Jupyter Notebook instance needs to decrypt it. This decryption is facilitated by the AWS KMS service using the same key that was used for encryption. For the decryption to occur successfully, the IAM role associated with the Jupyter Notebook instance must have a policy that includes 'kms:Decrypt' permission for the used KMS key. Without this permission, the Jupyter Notebook instance would be unable to read the data stored in the S3 bucket. Download a KMS-Encrypted Object from Amazon S3 (https://aws.amazon.com/premiumsupport/knowledge-center/decrypt-kms-encrypted-objects-s3/)"
  },
  {
    Question: "You have been tasked with transforming highly sensitive data using AWS Glue. Which of the following AWS Glue settings allowing you to control encryption for your transformation process?",
    OptionA: "Encrypting the managed EBS volumes used to run Apache Spark environment running PySpark code",
    OptionB: "Encryption of the classifier used during the transformation job",
    OptionC: "Encryption of your Data Catalog at its components using symmetric keys",
    OptionD: "The server-side encryption setting (SSE-S3 or SSE-KMS) that is passed as a parameter to your AWS Glue ETL job.",
      CorrectAnswer: "Encryption of your Data Catalog at its components using symmetric keys",
   
    Description: "You can enable encryption of your AWS Glue Data Catalog objects in the Settings of the Data Catalog on the AWS Glue console. You can enable or disable encryption settings for the entire Data Catalog. In the process, you specify an AWS KMS key that is automatically used when objects, such as tables, are written to the Data Catalog. AWS Glue supports only symmetric customer master keys (CMKs). The AWS KMS key list displays only symmetric keys. However, if you select Choose a KMS key ARN, the console lets you enter an ARN for any key type. Ensure that you enter only ARNs for symmetric keys. (https://docs.aws.amazon.com/glue/latest/dg/encrypt-glue-data-catalog.html)[https://docs.aws.amazon.com/glue/latest/dg/encrypt-glue-data-catalog.html]"
  },
  {
    Question: "A machine learning specialist is working with a dataset to create a model using a supervised learning algorithm. The specialist initially splits the data into two different sets, one for training and reserves the other for testing. The ratio of the dataset is broken into 80% training and 20% testing. After training the model the evaluation is yielding odd results. The model is 95% accurate for the training data and 64% accurate for the testing data. What is the reason for these odd results, and what action needs to be taken to resolve issue?",
    OptionA: "The model is currently underfitting on the dataset and does not know how to generalize for new data seen. This can be fixed by changing the hyperparameters to make the model (plus/minus the learning rate), then retrain.",
    OptionB: "The model is currently underfitting on the dataset and does not know how to generalize for new data seen. The underfitting of the model can be fixed by changing the hyperparameters to make the model (plus/minus the number of epochs), then retrain.",
    OptionC: "The testing dataset has highly imbalanced labels. Reshuffle the data more evenly across both training and testing datasets.",
    OptionD: "You need to use accuracy as the objective metric for the training set and use a different objective metric to measure accuracy on the training set (F1, Precision, Recall) to avoid the odd result.",
    CorrectAnswer: "The testing dataset has highly imbalanced labels. Reshuffle the data more evenly across both training and testing datasets.",
       Description: "This is a classic problem of data imbalancement meaning there are many observations of one type and not many of the other types. This is because we did not randomize the data before splitting it. We can redistribute the data and retrain the model. We could also try K-Fold cross validation. Validate a Machine Learning Model - Amazon SageMaker"
  },
  {
    Question: "You have JSON data that needs to be streamed into S3 in parquet format. How could you do this using the least effort?",
    OptionA: "Use Kinesis Firehose as delivery stream. Enable record transformation that references a table stored in Apache Hive metastore in EMR.",
    OptionB: "Setup EMR cluster that uses Apache Streaming to stream data onto cluster. Create an Apache Spark job to convert the JSON to parquet format using an Apache Hive metastore to determine the schema of the JSON data.",
    OptionC: "Use Kinesis Data Stream to ingest the data and Kinesis Data Firehose as a delivery stream. Once data is uploaded to S3, trigger a Lambda function that converts the data from JSON to parquet format.",
    OptionD: "Use Kinesis Firehose as delivery stream. Enable record transformation that references a table stored in AWS Glue defining the schema for your source records.",
      CorrectAnswer: "Use Kinesis Firehose as delivery stream. Enable record transformation that references a table stored in AWS Glue defining the schema for your source records.",
   
    Description: "Amazon Kinesis Data Firehose is the easiest way to load streaming data into AWS. Kinesis Data Firehose delivery stream can automatically convert the JSON data into Apache Parquet or Apache ORC format before delivering it to your S3 bucket. Kinesis Data Firehose references table definitions stored in AWS Glue. Choose an AWS Glue table to specify a schema for your source records. Stream Real-Time Data in Apache Parquet or ORC Format Using Amazon Kinesis Data Firehose"
  },
  {
    Question: "You are working for an online shopping platform that records actions made by its users. This information is captured in multiple JSON files stored in S3. You have been tasked with moving this data into Amazon Redshift database tables as part of a data lake migration process. Which of the following needs to occur to achieve this in the most efficient way?",
    OptionA: "Launch an Amazon Redshift cluster and create database tables.",
    OptionB: "Use the INSERT command to load the tables from the data files on Amazon S3.",
    OptionC: "Troubleshoot load errors and modify your COPY commands to correct the errors.",
    OptionD: "Use COPY commands to load the tables from the data files on Amazon S3.",
   
    CorrectAnswer: "Launch an Amazon Redshift cluster and create database tables.",
    
    Description: "You can add data to your Amazon Redshift tables either by using an INSERT command or by using a COPY command. At the scale and speed of an Amazon Redshift data warehouse, the COPY command is many times faster and more efficient than INSERT commands. You can load data from an Amazon DynamoDB table, or from files on Amazon S3, Amazon EMR, or any remote host through a Secure Shell (SSH) connection. When loading data from S3, you can load table data from a single file, or you can split the data for each table into multiple files. The COPY command can load data from multiple files in parallel. Using a COPY Command to Load Data - Amazon Redshift"
  },
  {
    Question: "You are a machine learning specialist that needs to setup an ETL pipeline for your organization using Amazon Elastic Map Reduce (EMR). You must connect the EMR cluster to Amazon SageMaker without writing any specific code. Which framework allows you to achieve this?",
    OptionA: "Apache Spark",
    OptionB: "Apache Hive",
    OptionC: "Apache Flink",
    OptionD: "Apache Pig",
  
    CorrectAnswer: "Apache Spark",
       Description: "Apache Spark can be used as an ETL tool to pre-process data and then integrate it directly with Amazon SageMaker for model training and hosting. Use Apache Spark with Amazon SageMaker - Amazon SageMaker"
  },
  {
    Question: "A major police station is needing to track parking in the downtown area of the city. The station is wanting to ingest videos of the cars parking in near-real time, use machine learning to identify license plates, and store that data in an AWS data store. Which solution meets these requirements with the least amount of development effort?",
    OptionA: "Use Amazon Kinesis Firehose to ingest the video in near-real time, and outputs results onto S3. Setup a Lambda function that triggers when a new video is PUT onto S3 to send results to Amazon Rekognition to identify license plate information, and then store results in DynamoDB.",
    OptionB: "Use Amazon Kinesis Data Streams to ingest the video in near-real time. Use the Kinesis Data Streams consumer integrated with Amazon Rekognition Video to process the license plate information, and then store results in DynamoDB.",
    OptionC: "Use Amazon Kinesis Data Streams to ingest videos in near-real time. Call Amazon Rekognition to identify license plate information, and then store results in DynamoDB.",
    OptionD: "Use Amazon Kinesis Video Streams to ingest the videos in near-real time. Integrate Kinesis Video Streams with Amazon Rekognition Video to identify the license plate information, then store the results in DynamoDB.",
    CorrectAnswer: "Use Amazon Kinesis Video Streams to ingest the videos in near-real time. Integrate Kinesis Video Streams with Amazon Rekognition Video to identify the license plate information, then store the results in DynamoDB.",
       Description: "Kinesis Video Streams is used to stream videos in near-real time. Amazon Rekognition Video uses Amazon Kinesis Video Streams to receive and process a video streams. After the videos have been processed by Rekognition we can output the results in DynamoDB Working with Streaming Videos - Amazon Rekognition"
  },
  {
    Question: "You are training a model using a dataset with credit card numbers stored in Amazon S3. What should be done to ensure these credit cards are encrypted before and during model training?",
    OptionA: "Create a SageMaker notebook instance with an SSE-KMS key associated with it. After loading the S3 data onto the notebook instance, encrypt it using SSE-KMS before feeding it into the training job.",
    OptionB: "When calling the SageMaker SDK training job, ensure the SSE-KMS is used as a parameter during the creation of the training job.",
    OptionC: "Create a Lambda function that is invoked when the training job starts to apply SSE-KMS key to the data before starting the training process.",
    OptionD: "Ensure the S3 bucket and data have an SSE-KMS key associated with it, and specify the same SSE-KMS Key ID when you create the SageMaker notebook instance and training job.",
    CorrectAnswer: "Ensure the S3 bucket and data have an SSE-KMS key associated with it, and specify the same SSE-KMS Key ID when you create the SageMaker notebook instance and training job.",
      Description: "Specify a KMS Key ID when you create Amazon SageMaker notebook instances, training jobs or endpoints. The attached ML storage volumes are encrypted with the specified key. You can specify an output Amazon S3 bucket for training jobs that is also encrypted with a key managed with KMS, and pass in the KMS Key ID for storing the model artifacts in that output S3 bucket. AWS KMS-based Encryption is Now Available in Amazon SageMaker Training and Hosting"
  },
  {
    Question: "You work for a company that is spinning up a new machine learning team. You need to set up a machine learning environment that will be accessed by multiple data scientists. Each new data scientist on the team needs to have their own Jupyter Notebook instance in Amazon SageMaker. You expect the team to scale rapidly. How should you manage access to SageMaker Notebook instances?",
    OptionA: "Allow the data scientists to assume an IAM role that grants them access only to their personal SageMaker notebook instances.",
    OptionB: "Ensure that each IAM policy associated with the data scientist's role and their respective notebook instance has the iam:PassRole denied.",
    OptionC: "Set up an ACL (Access Control List) for each notebook instance. Attach each data scientist role to the ACL associated with their personal notebook instance.",
    OptionD: "Use Amazon CloudWatch to trigger a Lambda function that restricts unauthorized access.",
       CorrectAnswer: "Allow the data scientists to assume an IAM role that grants them access only to their personal SageMaker notebook instances.",
       Description: "For heightened security and scalability, it's advisable to use IAM roles in this scenario. We can set up an IAM role that each data scientist can assume, granting them access only to their respective notebook instances in Amazon SageMaker. This way, we avoid distributing long-term AWS security credentials, offer temporary access, and can easily scale by integrating with existing identity systems. While IAM roles introduce an extra step of assuming a role, the security and scalability benefits in a growing organization make it a preferable choice. Reference: Step 7: Create an IAM Role for Amazon SageMaker Notebooks - AWS Glue"
  },
  {
    Question: "You have a data set with one column missing 30% of its data. You notice that the missing features can be determined from other features in the data set. What can you do to replace the values that will cause least amount of bias?",
    OptionA: "use mean value",
    OptionB: "multiple data imputations",
    OptionC: "last observed carried forward",
    OptionD: "removing the items with missing values",
    CorrectAnswer: "multiple data imputations",
   
    Description: "Multiple imputation for missing data makes it possible for the researcher to obtain approximately unbiased estimates of all the parameters from the random error. The researcher cannot achieve this result from deterministic imputation, which the multiple imputation for missing data can do. Multiple Imputation for Missing Data: Concepts and New Development"
  },
  {
    Question: "A term frequency–inverse document frequency (tf–idf) matrix using trigrams is built from a text corpus consisting of the following three documents: { Hold please }, { Please try again }, { Please call us back }. What are the dimensions of the tf–idf vector/matrix?",
    OptionA: "(9, 3)",
    OptionB: "(3, 16)",
    OptionC: "(3, 9)",
    OptionD: "(3, 3)",
       CorrectAnswer: "(3, 16)",
  
    Description: "You can determine the tf-idf vectorized format by using the following: (number of documents, number of unique n-grams). There are 3 documents (or corpus data we are vectorizing) with 3 unique trigrams ['call us back', 'please call us', 'please try again'], 6 unique bigrams ['call us', 'hold please', 'please call', 'please try', 'try again', 'us back'], and 7 unique unigrams ['again', 'back', 'call', 'hold', 'please', 'try', 'us']. So, the vectorized matrix is (3, 16). TfidfVectorizer."
  },
  {
    Question: "Your organization has the need to set up a petabyte scaled BI and dashboard analysis tool that will query millions of rows of data spread across thousands of files stored in S3. Your organization wants to save as much money as possible. Which solution will allows developers to run dozens if not hundreds or thousands of queries per day, and possibly scanning many TBs of data each, while still being cost effective?",
    OptionA: "EC2 Spot instances and Presto",
    OptionB: "AWS Glue Data Catalog and Amazon Athena",
    OptionC: "Data Pipeline and RDS",
    OptionD: "Lambda Functions with extremely long timeouts",
    CorrectAnswer: "EC2 Spot instances and Presto",
   
    Description: "You pay a constant fee for the compute instances you are running (EC2 instances cost). The more machines you run and the bigger they are - the higher the fee, yes. But Presto is very efficient and if your data is correctly stored, a few commodity machines will do a great job if you are running your Presto cluster on the same region as your S3 bucket, and within one AZ, as there is no network or data transfer costs at all. The compute costs can be further optimized by using spot instances for worker nodes, and completely shutting them down off-hours (where applicable). Presto can deal with a lost worker node - which might slow down some queries but spot instances come at a great discount. Presto - Amazon EMR Presto | Distributed SQL Query Engine for Big Data"
  },
  {
    Question: "You work for a team that has a model being used in production, for which the data it is sent to perform inferences on is coming from a different source. The model was built to work well for cleaned data inputs. How do you ensure that the model’s performance in production will be similar?",
    OptionA: "Ensuring that the data is accurate for data inputs and training data.",
    OptionB: "Never allow input data for a production model come from another data source.",
    OptionC: "Create a Lambda function that replaces missing values with the mean value on the data source before it is used in production.",
    OptionD: "Review counts, data durations, and the precision of the data inputs compared to training data.",
   
    CorrectAnswer: "Ensuring that the data is accurate for data inputs and training data.",
   
    Description: "Comparing counts lets you identify, track, and highlight data loss, and test against what seems reasonable. Reviewing data duration lets you determine what time period each dataset is for. Quantify precision by comparing the mean, median and standard deviation of the data source and the data used to train the model. Calculate the number or percentage of outliers. For lower dimensional data or key variables, boxplots can provide a quick visual assessment of reasonableness."
  },
  {
    Question: "You are a machine learning specialist analyzing a large dataset with 20 features. What visualization can be used to show all 20 features and the correlation they have with all other features?",
    OptionA: "Histogram",
    OptionB: "Bubble Chart",
    OptionC: "Bar Chart",
    OptionD: "Box Plot",
       CorrectAnswer: "Heatmap",
      Description: "Heatmaps are a great way to show correlation and compare values to other values (see how one value affects the other). Using Heat Maps - Amazon QuickSight"
  },
  {
    Question: "You are applying normalization techniques to a column in your dataset. The column has the following values {1, 5, 7}. When we apply normalization what will the respective output results be?",
    OptionA: "{0.00, 0.66, 1.00}",
    OptionB: "{0.00, -0.66, 1.00}",
    OptionC: "{-1.00, 0.66, 1.00}",
    OptionD: "{-1.33, 0.26, 1.06}",
     CorrectAnswer: "{0.00, 0.66, 1.00}",
   
    Description: "Applying normalization translates each feature individually such that it is in the given range on the training set between 0 and 1. In this case {1, 5, 7} maps to {0.00, 0.66, 1.00} respectively. sklearn.preprocessing.MinMaxScaler — scikit-learn 0.21.2 documentation"
  },
  {
    Question: "You work for an organization that handles highly sensitive information on a daily basis. The company has different compliance rules that require all data be encrypted at rest. When preparing your machine learning models using SageMaker, how can you achieve these requirements?",
    OptionA: "Stop your SageMaker notebook instance, create a customer managed key in KMS and attach it to the stopped SageMaker Notebook instance",
    OptionB: "Unmount the EBS volume from SageMaker Notebook instance, encrypt it with a KMS key, and reattach to SageMaker Notebook instance",
    OptionC: "Create a customer managed key in KMS and use it when creating your SageMaker Notebook instance",
    OptionD: "Ensure the role associated with the SageMaker Notebook instance is assigned to the customer managed key in KMS",
    CorrectAnswer: "Create a customer managed key in KMS and use it when creating your SageMaker Notebook instance",
    
    Description: "You can encrypt your Amazon SageMaker storage volumes used for Training and Hosting with AWS Key Management Service (KMS). AWS KMS gives you centralized control over the encryption keys used to protect your data. You can create, import, rotate, disable, delete, define usage policies for, and audit the use of encryption keys used to encrypt your data. You specify a KMS Key ID when you create Amazon SageMaker notebook instances, training jobs or endpoints. The attached ML storage volumes are encrypted with the specified key. AWS KMS-based Encryption is Now Available in Amazon SageMaker Training and Hosting"
  },
  {
    Question: "You have been tasked with creating a labeled dataset by classifying text data into different categories depending on the summary of the corpus. You plan to use this data with a particular machine learning algorithm within AWS. Your goal is to make this as streamlined as possible with minimal amount of setup from you and your team. What tool can be used to help label your dataset with the minimum amount of setup?",
    OptionA: "AWS Comprehend entity detection",
    OptionB: "Marketplace AMI for NLP problems",
    OptionC: "AWS Comprehend sentiment analysis",
    OptionD: "Amazon Neural Topic Modeling (NTM) built-in algorithm",
       CorrectAnswer: "AWS SageMaker GroundTruth text classification job",
     Description: "You can use SageMaker Ground Truth to create ground truth datasets by creating labeling jobs. When you create a text classification job, workers group text into the categories that you define. You can define multiple categories but the worker can apply only one category to the text. Use the instructions to guide your workers to make the correct choice. Always define a generic class in addition to your specific classes. Giving your workers a generic option helps to minimize inaccurately classified text. Amazon SageMaker Ground Truth - Amazon SageMaker"
  },
  {
    Question: "You have been given a dataset and are in the stages of analyzing it. This dataset has around 200 features in both numeric and categorical formats, many of which contain incomplete values. You decide to perform dimensionality reduction on the dataset hoping it will help create a more robust machine learning model. Which of the following techniques would perform better for reducing dimensions of our dataset?",
    OptionA: "Removing columns which have more than 95% many missing values",
    OptionB: "Removing columns with dissimilar data trends",
    OptionC: "Using the cartesian product of different features to create more relevant features",
    OptionD: "Removing columns which have high variance in data",
    CorrectAnswer: "Removing columns which have more than 95% many missing values",
       Description: "If the features are missing the majority of values, say 95%+, we can remove such columns. When you have 100s or even 1000s of features, Dimension Reduction is generally good to consider. Dimensionality reduction - Wikipedia"
  },
  {
    Question: "A machine learning specialist needs to create a model for detecting whether a person is sleeping or meditating. The dataset consists of different features of brainwaves that have been collected over a 5 year span. Since true meditation is very difficult to achieve the data is highly imbalanced. Less than 2% of the dataset was labeled as true meditation. Which solution provides the optimal predictive power for classifying a person as sleeping or meditating?",
    OptionA: "Resample the dataset using oversampling/undersampling and use the F beta (F1) score as the objective metric. Finally, apply the XGBoost algorithm.",
    OptionB: "Oversample the dataset using a clustering technique and use accuracy as the objective metric. Finally, apply Random Cut Forest (RCF) algorithm.",
    OptionC: "Undersample the majority class in the dataset using a cluster technique and use precision as the objective metric. Finally, apply the Random Cut Forest algorithm.",
    OptionD: "Resample the dataset using oversample/undersampling and use accuracy as the objective metric. Finally, apply the XGBoost algorithm.",
    CorrectAnswer: "Resample the dataset using oversampling/undersampling and use the F beta (F1) score as the objective metric. Finally, apply the XGBoost algorithm.",
 
    Description: "When the data is highly imbalanced, using accuracy as the objective metric is not a good idea. Instead using precision, recall, or F1 score is a better option. Since this is a binary classification problem, the XGBoost is a great algorithm for this. Random Cut Forest (RCF) is used for anomaly detection. XGBoost Algorithm - Amazon SageMaker Create a model for predicting orthopedic pathology using Amazon SageMaker | AWS Machine Learning Blog"
  },
  {
    Question: "You are preparing plain text corpus data to build a model for Amazon's Neural Topic Model (NTM) algorithm. What are the steps you need to take before the data is ready for training?",
    OptionA: "First tokenize the corpus data. Then, count the occurrence of each token and form bag-of-words vectors. Use these vectors as training data.",
    OptionB: "First perform tf-idf to remove words that are not important. Use the number of unique n-grams to create vectors and respective word counts. Use these vectors as training data.",
    OptionC: "First normalize the corpus data. Then, count the occurrence of each of the value produced, creating word count vectors. Use these vectors as training data.",
    OptionD: "First create bigrams of the corpus data. Then, count the occurrence of each bigram produced, creating word count vectors. Use these vectors as training data.",
    CorrectAnswer: "First tokenize the corpus data. Then, count the occurrence of each token and form bag-of-words vectors. Use these vectors as training data.",
       Description: "Both in training and inference, need to be vectors of integers representing word counts. This is so-called bag-of-words (BOW) representation. To convert plain text to BOW, we need to first “tokenize” our documents, that is, identify words and assign an integer ID to each of them. Then, we count the occurrence of each of the tokens in each document and form BOW vectors. Introduction to the Amazon SageMaker Neural Topic Model | AWS Machine Learning Blog"
  },
  {
    Question: "You are working with several scikit-learn libraries to preprocess and prepare your data. You also have created a script that trains your model using scikit-learn. You have been tasked with using SageMaker to train your model using this custom code. What can be done to run scikit-learn jobs directly in Amazon SageMaker?",
    OptionA: "Include your training script within a Notebook instance on Amazon SageMaker. Install scikit-learn inside a Docker container that run your script. Upload container to ECR and use within Amazon SageMaker notebook instance.",
    OptionB: "Upload your training script to Amazon S3. Use a Notebook instance in Amazon SageMaker to run the code from whatever instance type you need.",
    OptionC: "Include your training script within a Notebook instance on Amazon SageMaker. Construct a sagemaker.sklearn.estimator.sklearn estimator. Train the model using the pre-build container provided by the Estimator.",
    OptionD: "Upload your training script to a Deep Learning AMI with scikit-learn pre-installed. Use Deep Learning AMI to train your model.",
    CorrectAnswer: "Include your training script within a Notebook instance on Amazon SageMaker. Construct a sagemaker.sklearn.estimator.sklearn estimator. Train the model using the pre-build container provided by the Estimator.",
      Description: "You can run and package scikit-learn jobs into containers directly in Amazon SageMaker. Feature Processing with Spark ML and Scikit-learn - Amazon SageMaker"
  },
  {
    Question: "You are apply standardization techniques to a feature in your dataset. The column has the following values {5, 20, 15}. The standard deviation is 6.23 and the mean of the feature 13.33. When we apply standardization what will the respective output results be?",
    OptionA: "{0, 0.66, 1}",
    OptionB: "{1.33, 1.06, 0.26}",
    OptionC: "{-1.33, 1.06, 0.26}",
    OptionD: "{0, 1, 0.66}",
     CorrectAnswer: "{-1.33, 1.06, 0.26}",
       Description: "Let's take the value 5. To calculate the standardization value we use the following formula z = (x - u) / s where 'z' is the standardized value, where 'x' is our observed value, where 'u' is the mean value of the feature, and 's' is the standard deviation. For 5, -1.33 = (5 - 13.33) / 6.23. For 15, 0.26 = (15 - 13.33) / 6.23. For 20, 1.06 = (20 - 13.33) / 6.23. Since 5 is the only value that produces a negative value, {-1.33, 1.06, 0.26} is the only acceptable answer. sklearn.preprocessing.StandardScaler — scikit-learn 0.21.2 documentation"
  },
  {
    Question: "You are a machine learning specialist designing a regression model to predict the sales for an upcoming summer sale. The data from the past sales consists of 1,000 records containing 20 numeric attributes. As you start to analyze the data, you discovered that 30 records have values that are above the top upper whisker in the box plot upper quartile. You confirm with management that these records are unusual, but certainly valid values. There are also 78 records where another numerical value is blank. What should you do to correct these problems?",
    OptionA: "Use the unusual data and replace the missing values with a separate Boolean variable",
    OptionB: "Drop the unusual records and replace the blank values with separate Boolean values",
    OptionC: "Drop the unusual records and fill in the blank values with 0",
    OptionD: "Drop the unusual records and replace the blank values with the mean value",
    CorrectAnswer: "Drop the unusual records and replace the blank values with the mean value",
        Description: "There are many different ways to handle this scenario. We can eliminate the answer that deals with creating a separate Boolean. This leaves the two answers with filling in the missing values with 0 or the mean. The mean is going to give us much better results than using 0. We should drop the unusual values and replace the missing values with the mean. Feature engineering - Wikipedia"
  },
  {
    Question: "You are working with a machine learning team training an image classification model using MXNet on Amazon SageMaker. The requirements state that the model should be at least 85% accurate. The data appears to be of good quality, but the accuracy is around 48% during training with the test data. Most of the time wrong labels are being predicted. What should be done to help increase the accuracy of the model?",
    OptionA: "Use Amazon SageMaker's automatic model tuning. Specify the objective metric and take the best performing parameters suggested by the service to use when training the model",
    OptionB: "Use Amazon SageMaker's automatic model tuning. Take the best performing hyperparameters and manually adjust them to meet your requirements.",
    OptionC: "Use Amazon SageMaker's automatic model tuning. Take the best performing hyperparameters and run multiple training jobs in parallel using Apache Spark and Spark ML",
    OptionD: "Use Amazon SageMaker's automatic model tuning. Use AWS Batch to run multiple batches of the training data with different hyper parameters specified during the autotuning job.",
    CorrectAnswer: "Use Amazon SageMaker's automatic model tuning. Specify the objective metric and take the best performing parameters suggested by the service to use when training the model",
       Description: "A Hyperparameter Tuning job launches multiple training jobs, with different hyperparameter combinations, based on the results of completed training jobs. Amazon SageMaker Automatic Model Tuning: Using Machine Learning for Machine Learning | AWS News Blog How Hyperparameter Tuning Works - Amazon SageMaker"
  },
  {
    Question: "How might you frame an ML approach for predicting if a random number generator will return a 1 or 0 given that the number generator is perfectly random and you are provided with 1000 prior number selections?",
    OptionA: "Logical Regression",
    OptionB: "Binary Classification",
    OptionC: "Machine Learning not needed",
    OptionD: "Forecasting",
       CorrectAnswer: "Machine Learning not needed",
   
    Description: "In this example, we are told that the random 0 or 1 generator is perfectly random, so the chances of the next number being 1 is exactly 50% regardless of the past 1000 selections. There is no need for a machine learning model in this case. Formulating the Problem - Amazon Machine Learning"
  },
  {
    Question: "You have progressively tested increasing the learning rate of your model. You get to a test where you observe that you reached convergence rather quickly then loss flattened out. You suspect your model is not as accurate as it could be due to its higher error rate. Which of the following is your most likely next step?",
    OptionA: "Add additional CPU instances for training.",
    OptionB: "Increase the learning rate and retrain.",
    OptionC: "Switch to GPU instances for training",
    OptionD: "Lower the learning rate and retrain.",
    CorrectAnswer: "Lower the learning rate and retrain.",
      Description: "Lowering the learning rate will allow the training process to make smaller adjustments, potentially allowing it to converge on a better, more accurate place. Learning rate - Wikipedia"
  },
  {
    Question: "A financial institution is seeking a way to improve security by implementing two-factor authentication (2FA). However, management is concerned about customer satisfaction by being forced to authenticate via 2FA for every login. The company is seeking your advice. What is your recommendation?",
    OptionA: "Create a ML model that uses IP Insights to detect anomalies in client activity. Only if anomalies are detected, force a 2FA step.",
    OptionB: "Create an ML model using Linear Learner that can evaluate whether a customer is truly a human or some scripted bot typical of hacking attempts. Hold off on implementing 2FA until there is sufficient data to support its need.",
    OptionC: "Create a binary classifier model using Object2Vec to detect unusual activity for customer logins. If unusual activity is detected, trigger an SNS notification to the Fraud Department.",
    OptionD: "Recommend that the company invests in customer education on why 2FA is important to their well-being. Train customer support staff on properly handling customer complaints.",
   
    CorrectAnswer: "Create a ML model that uses IP Insights to detect anomalies in client activity. Only if anomalies are detected, force a 2FA step.",
   
    Description: "IP Insights is a built-in SageMaker algorithm that can detect anomalies as it relates to IP addresses. In this case, only enforcing 2FA where unusual activity is detected might be a good compromise between security and ease-of-use. While using facial recognition might be a tempting alternative, it can easily be bypassed by holding up a picture of some customer and it would not be true multi-factor authentication. IP Insights Algorithm - Amazon SageMaker"
  },
  {
    Question: "You are designing an image classification model that will detect objects in provided pictures. Which neural network approach would be most likely in this use case?",
    OptionA: "Convolutional Neural Network",
    OptionB: "Decepticon Neural Network",
    OptionC: "Recurrent Neural Network",
    OptionD: "Stochastic Neural Network",
       CorrectAnswer: "Convolutional Neural Network",
   
    Description: "Convolutional Neural Networks are most commonly associated with image and signal processing. Recurrent Neural Networks are most commonly used with text or speech use-cases where sequence prediction is key. When to Use MLP, CNN, and RNN Neural Networks"
  },
  {
    Question: "In AWS SageMaker automatic hyperparameter tuning, which of the following methods are supported?",
    OptionA: "Stochastic search",
    OptionB: "Hyperband",
    OptionC: "Random search",
    OptionD: "Grid search",
      CorrectAnswer: "Hyperband",
    
    Description: "Hyperband is a dynamic tuning strategy in SageMaker that reallocates resources based on the performance of training jobs. It combines elements of other methods, using both intermediate and final results to prioritize promising hyperparameter configurations. This approach aims to achieve a balance between computational efficiency and the accuracy of the tuned model."
  },
  {
    Question: "You are working with a colleague in your company on creating some documentation for using SageMaker internally with XGBoost. Your colleague is based in France while you are based in Canada. You are reviewing her documentation and notice that the training image registry path does not match the path that you have recorded in your version. What is the most likely reason for this?",
    OptionA: "Built-in algorithm registry paths are randomly generated for each account.",
    OptionB: "You selected a training image while your colleague selected an inference image.",
    OptionC: "Your colleague has selected the wrong version of XGBoost.",
    OptionD: "You both are using different regions.",
    CorrectAnswer: "You both are using different regions.",

    Description: "The registry paths for the training and inference images for built-in algorithms differ by region. It is likely that you and your colleague are using different regions to prepare your documentation. Common Parameters for Built-In Algorithms - Amazon SageMaker"
  },
  {
    Question: "You have build two slightly different models for performing multi-class classification. What metric would be the most holistic for evaluating the models against each other?",
    OptionA: "Precision",
    OptionB: "Macro F1 Score",
    OptionC: "Recall",
    OptionD: "k-Fold Cross-validation",
       CorrectAnswer: "Macro F1 Score",
      Description: "The Macro F1 Score is an unweighted average of the F1 score across all classes and is typically used to evaluate the accuracy of multi-class models. A number closer to 1 indicates higher accuracy. Multiclass Model Insights - Amazon Machine Learning"
  },
  {
    Question: "You have developed a very complex deep learning model, but your accuracy levels are still not at your desired target levels, even after hyperparameter optimization. What is the most likely cause?",
    OptionA: "You did not employ a warm start method for your first optimization job.",
    OptionB: "Hyperparameter tuning is not flawless and can still fail to converge on the best answer.",
    OptionC: "Hyperparameter optimization is not performing as documented.",
    OptionD: "A Bayesian approach is being used instead of a random approach for hyperparameter optimization.",
    CorrectAnswer: "Hyperparameter tuning is not flawless and can still fail to converge on the best answer.",
     Description: "Hyperparameter tuning can accelerate your productivity by trying many variations of a model, focusing on the most promising combinations of hyperparameter values within the ranges that you specify. However, it's still possible that the tuning job will fail to converge on the best answer, even if the best possible combination of values is within the range that you choose. You should not over-rely on automatic tuning for optimization but rather include it as part of a scientific approach. Reference: How Hyperparameter Tuning Works - Amazon SageMaker"
  },
  {
    Question: "A company is building an application that allows high school students to view programming videos to learn more about coding. The instructors upload videos directly to the platform. You have been tasked with designing a model to determine whether the videos uploaded are safe for viewing by high school students. It is critical that no inappropriate videos make it onto the platform. This means a positive case would be marking an unsafe video as unsafe (true positive). Which is the MOST important metric to evaluate during the machine learning process for this task?",
    OptionA: "Recall",
    OptionB: "AUC/ROC",
    OptionC: "Precision",
    OptionD: "Accuracy",
    CorrectAnswer: "Recall",
  
    Description: "The most important metric to evaluate is going to be the recall metric. Since it's extremely important we find all the explicit cases, then this will be our positive class. If the model predicts a video is NOT explicit and the video is explicit, this is the most expensive case. We want to minimize these. This means we want to minimize the False Negatives, which makes Recall the most important metric. Precision and recall - Wikipedia"
  },
  {
    Question: "You are designing a machine learning model to dynamically translate from a variety of languages to Klingon. What algorithm might be the best approach for this use-case?",
    OptionA: "NTM",
    OptionB: "BlazingText",
    OptionC: "AWS Translate",
    OptionD: "Seq2Seq",
     CorrectAnswer: "Seq2Seq",
    
    Description: "For use cases that involve converting a sequence of tokens (or words in this case) to another set of tokens, Seq2Seq would be the best algorithm choice. Seq2seq is often used for machine translation of language. Unfortunately, Amazon Translate does not support Klingon at the moment. Sequence-to-Sequence Algorithm - Amazon SageMaker"
  },
  {
    Question: "You are preparing a large set of CSV data for a training job using K-Means. Which of the following are NOT actions that you should expect to take in this scenario?",
    OptionA: "Decide on the number of clusters you want to target.",
    OptionB: "Decide on the value you want to assign to k.",
    OptionC: "Use a mean or median strategy to populate any missing label data.",
    OptionD: "Convert the data to protobuf RecordIO format.",
   
    CorrectAnswer: "Use a mean or median strategy to populate any missing label data.",
    
    Description: "K-Means is an unsupervised clustering algorithm and does not use labels that might be associated with the objects in the training dataset. How K-Means Clustering Works - Amazon SageMaker"
  },
  {
    Question: "A machine learning specialist is running a training job on a single EC2 instance using their own Tensorflow code on a Deep Learning AMI. The specialist wants to run distributed training and inference using SageMaker. What should the machine learning specialist do?",
    OptionA: "Use Tensorflow in SageMaker and modify the AWS Deep Learning Docker containers",
    OptionB: "Use Tensorflow in SageMaker and run your code as a script",
    OptionC: "It is not possible to run custom Tensorflow code in SageMaker",
    OptionD: "Use Tensorflow in SageMaker and edit your code to run using the SageMaker Python SDK",
       CorrectAnswer: "Use Tensorflow in SageMaker and edit your code to run using the SageMaker Python SDK",
   
    Description: "When using custom TensorFlow code, the Amazon SageMaker Python SDK supports script mode training scripts. Script mode has the following advantages: Script mode training scripts are more similar to training scripts you write for TensorFlow in general, so it is easier to modify your existing TensorFlow training scripts to work with Amazon SageMaker. Script mode supports both Python 2.7- and Python 3.6-compatible source files. Script mode supports Horovod for distributed training. Use TensorFlow with Amazon SageMaker - Amazon SageMaker"
  },
  {
    Question: "After a training and testing session, you notice that your training accuracy is 98% while you accuracy during testing was only 67%. What might you do to improve the model?",
    OptionA: "Rerun the training process with a larger learning rate.",
    OptionB: "Reduce the number of features being analyzed in the model.",
    OptionC: "Change the approach from Linear Regression to Logistic Regression.",
    OptionD: "Ensure the data was properly randomized before the split.",
      CorrectAnswer: "Reduce the number of features being analyzed in the model.",
   
    Description: "When training accuracy is high and testing accuracy is low, it usually indicates overfitting or insufficient randomization across the training and testing datasets. You can randomize the data and try training again or maybe use a cross-validation method like k-fold. For overfitting, one suggestion is to reduce the number of features in the model. Model Fit: Underfitting vs. Overfitting - Amazon Machine Learning"
  },
  {
    Question: "Which of the following is NOT a valid use-case for incremental training?",
    OptionA: "Use the model artifacts or a portion of the model artifacts from a popular publicly available model in a training job. You don't need to train a new model from scratch.",
    OptionB: "Rebuilt model artifacts which you have accidentally deleted.",
    OptionC: "Train a new model using an expanded dataset that contains an underlying pattern that was not accounted for in the previous training and which resulted in poor model performance.",
    OptionD: "Train several variants of a model, either with different hyperparameter settings or using different datasets.",
      CorrectAnswer: "Rebuilt model artifacts which you have accidentally deleted.",
   
    Description: "Incremental training is used for all of these except rebuilding model artifacts. Incremental training picks up where another training job left off by using the existing artifacts created by the prior training job. Incremental Training in Amazon SageMaker."
  },
  {
    Question: "In a regression problem, we have plotted the residuals in a histogram and observed that a distribution is heavily skewed to the left of zero. What does this tell us about our model?",
    OptionA: "Our model is sufficient with regard to aggregate residual.",
    OptionB: "Our model is not perfect but still well within the area under the curve.",
    OptionC: "Our model is sufficient with regard to RMSE.",
    OptionD: "Our model is consistently overestimating.",
   
    CorrectAnswer: "Our model is consistently overestimating.",
    
    Description: "The residual is commonly defined as the actual value minus the predicted value. If most of our residuals are negative numbers, that means that our predicted values are mostly more than the actual values. This means that our model is consistently overestimating. Regression - Amazon Machine Learning"
  },
  {
    Question: "After training and validation sessions, you notice that the error rate is higher than expected for both sessions. What could you do to reduce the error rates for your model?",
    OptionA: "Run training for a longer period of time.",
    OptionB: "Add more variables to the dataset",
    OptionC: "Run a random cut forest algorithm on the data.",
    OptionD: "Run a kNN algorithm on the data.",
      CorrectAnswer: "Run training for a longer period of time.",
   
    Description: "When both training and testing errors are high, it indicates that our model is underfitting the data. We can try to add more details to the dataset, gather more data for training and/or run the training session longer. We might also need to identify a better algorithm. Train faster, more flexible models with Amazon SageMaker Linear Learner | AWS Machine Learning Blog"
  },
  {
    Question: "A binary classification model has been created to sort parts on an assembly line into acceptable (1) or unacceptable (0), based on a complex array of readings. The model incorrectly decides that some flawed parts are acceptable (1) when they should have been marked as unacceptable (0). Which of the following correctly defines this type of result?",
    OptionA: "True Negative",
    OptionB: "False Negative",
    OptionC: "False Positive",
    OptionD: "True Positive",
     CorrectAnswer: "False Positive",
  
    Description: "A false positive result is when a model inaccurately predicts a value at 1 (acceptable), but the true value is 0 (unacceptable). In this case, our model inaccurately passed through the flawed parts (true value of 0) with a predicted value of 1 for good parts, which makes this a false positive result. A false positive is also known as a Type I error. Type I and type II errors - Wikipedia AWS Binary Model Insights"
  },
  {
    Question: "Which of the following could be used in an API to estimate the value of a car?",
    OptionA: "Multi-class Classification",
    OptionB: "Binary Classification",
    OptionC: "Polynomial Synthesis",
    OptionD: "Linear Regression",
    CorrectAnswer: "Linear Regression",
       Description: "When assessing against a continuous value, such as a car price, linear regression will be the best option to assess the value and where it fits within market value Use Amazon SageMaker Built-in Algorithms - Amazon SageMaker"
  },
  {
    Question: "You are consulting with a large financial institution on a ML model using a built-in SageMaker algorithm. They have asked for help deciding which hyperparameters and ranges to use in an automatic model tuning job. What can you recommend to help them get started?",
    OptionA: "Use a Bayesian approach when choosing target parameters and recommended ranges.",
    OptionB: "Consult the documentation regarding the tunable parameters and recommended ranges.",
    OptionC: "Use a random approach when choosing parameters and recommended ranges.",
    OptionD: "All algorithm hyperparameters are available for auto-tuning but you must choose the proper target metric scale.",
 
    CorrectAnswer: "Consult the documentation regarding the tunable parameters and recommended ranges.",
      Description: "Not all algorithm metrics are able to be used as optimization metrics. You should consult the documentation on the specific algorithm to determine which hyperparameters are able to be tuned. Define Metrics - Amazon SageMaker"
  },
  {
    Question: "You are consulting for a logistics company who wants to implement a very specific algorithm for warehouse storage optimization. The algorithm is not part of the currently available SageMaker built-in algorithms. What are your options?",
    OptionA: "Use a series of existing algorithms to simulate the actions of the unavailable algorithm.",
    OptionB: "Build the algorithm in a docker container and use that custom algorithm for training and inference in SageMaker.",
    OptionC: "Wait until the algorithm is available in SageMaker before further work.",
    OptionD: "Post an incendiary message to Twitter hoping to shame AWS into adopting the specialized algorithm.",
       CorrectAnswer: "Build the algorithm in a docker container and use that custom algorithm for training and inference in SageMaker.",
    
    Description: "If SageMaker does not support a desired algorithm, you can either bring your own or buy/subscribe to an algorithm from the AWS Marketplace. Use Your Own Algorithms or Models with Amazon SageMaker - Amazon SageMaker"
  },
  {
    Question: "A ski equipment company is trying to predict the expected sales from a line of ski goggles. They have never sold this type of product before, but they do have some historic sales data for other products which they believe have similar market adoption curves. What would be your first algorithm of choice among the built-in SageMaker algorithms for this use-case?",
    OptionA: "DeepAR",
    OptionB: "Linear Learner",
    OptionC: "Factorization Machines",
    OptionD: "K-Nearest Neighbor",
       CorrectAnswer: "DeepAR",
   
    Description: "While there may be many ways to create a forecasting model, the SageMaker DeepAR Forecasting algorithm is most closely positioned for this use case. We can use multiple sets of historic data together to create a more refined forecast than if we were just using a single product sales history dataset. DeepAR Forecasting Algorithm - Amazon SageMaker"
  },
  {
    Question: "You have been brought in to help a Data Science group within a large manufacturing company migrate their existing ML processes to AWS. They currently use a pre-trained word vector model using fastText for text classification. The team is not satisfied with the performance of the existing model and are open to the possibility of either changing the model or other configuration changes. What would be the most efficient path using as much of the AWS platform as possible?",
    OptionA: "Deploy TensorFlow on EC2 spot instances to use the pre-trained model.",
    OptionB: "Deploy EMR with Mahout to use the fastText model.",
    OptionC: "Create a Docker container for the fastText algorithm and upload to the ECR.",
    OptionD: "Use the built-in algorithms provided by SageMaker to host the model.",
  
    CorrectAnswer: "Use the built-in algorithms provided by SageMaker to host the model.",

    Description: "The BlazingText algorithm built into SageMaker can host pre-trained fastText models. This would be the least complex way to migrate the pre-existing model to AWS. Train a Model with Amazon SageMaker - Amazon SageMaker"
  },
  {
    Question: "You need to organize a large set of data into 6 groupings that are more similar than dissimilar. What algorithm approach might you use for this problem? You also don't have any ground truth data available for this task.",
    OptionA: "Linear Regression",
    OptionB: "K-Nearest Neighbor",
    OptionC: "Multi-Class Classification",
    OptionD: "Random Cut Forest",
       CorrectAnswer: "K-Means",
       Description: "K-means is an unsupervised learning algorithm. It attempts to find discrete groupings within data, where members of a group are as similar as possible to one another and as different as possible from members of other groups. You define the attributes that you want the algorithm to use to determine similarity. K-Means Algorithm - Amazon SageMaker"
  },
  {
    Question: "Creating an S3 VPC Endpoint in your VPC will have which of the following impacts?",
    OptionA: "Reduce security.",
    OptionB: "Improve security.",
    OptionC: "Increase latency.",
    OptionD: "Reduce egress costs.",
   
    CorrectAnswer: "Improve security.",
   
    Description: "Using a VPC Endpoint will redirect the S3 traffic through the AWS private network rather than egressing to the public internet. Both of these attributes will reduce egress costs and increase security. VPC Endpoints - Amazon Virtual Private Cloud"
  },
  {
    Question: "You are consulting for a large intelligence organization that has very strict rules around how data must be handled. One such rule is that data cannot be allowed to transit the public internet. What might you suggest as they are setting up SageMaker Notebook instances?",
    OptionA: "VPC Interface Endpoints",
    OptionB: "AWS Macie",
    OptionC: "AWS CloudTrail",
    OptionD: "VPC Log Monitoring",
      CorrectAnswer: "VPC Interface Endpoints",
   
    Description: "VPC Interface Endpoints allow traffic to flow between a VPC and select AWS services without having to exit to the public internet. This would be a good way to keep the sensitive data off the internet. Connect to Amazon SageMaker Through a VPC Interface Endpoint - Amazon SageMaker"
  },
  {
    Question: "After several weeks of working on a model for genome mapping, you believe you have perfected it and now want to deploy it to a platform that will provide the highest performance. Which of the following AWS platforms will provide the highest cost-to-performance for this compute-intensive model?",
    OptionA: "EC2 X1 Instance",
    OptionB: "EC2 G3 Instance",
    OptionC: "EC2 M5 Instance",
    OptionD: "EC2 F1 instance",
      CorrectAnswer: "EC2 G3 Instance",
      Description: "EC2 G3 instances are designed for high-performance computing (HPC) and machine learning (ML) workloads. They feature powerful NVIDIA GPUs and Intel CPUs, making them ideal for genome mapping workloads. EC2 G3 instances are relatively inexpensive, making them a good choice for cost-sensitive projects."
  },
  {
    Question: "You are working on a project for a client that wants to collect metadata for speeches given at various medical conferences around the world. The company wants to extract key entities for a search engine targeted to an English-speaking population even though the speeches are given in the local language which may be French, Italian, English, German or Korean. What collection of services would best fit this use-case?",
    OptionA: "Amazon Transcribe -> Amazon Translate -> Amazon Comprehend",
    OptionB: "Amazon Lex -> Amazon Polly -> Amazon Transcribe",
    OptionC: "Amazon Transcribe -> Amazon Textract",
    OptionD: "Amazon Comprehend -> Amazon Translate -> Amazon Lex",
  
    CorrectAnswer: "Amazon Transcribe -> Amazon Translate -> Amazon Comprehend",
      Description: "For this use case, we first would want to transcribe the verbal speed into text using Amazon Transcribe. We can then translate the text to English using Amazon Translate. Once we have the text in English, we can extract topics using Amazon Comprehend. Machine Learning on AWS"
  },
  {
    Question: "You have setup autoscaling for your deployed model using SageMaker Hosting Services. You notice that in times of heavy load spikes, it takes a long time for the hosted model to scale out in response to the load. How might you speed up the autoscaling process?",
    OptionA: "Change the timeout in the auto-scaling Lambda function.",
    OptionB: "Disable CloudWatch advanced tracking metrics.",
    OptionC: "Change the scale metric from InvocationsPerInstance to MemoryUtilization.",
    OptionD: "Create a new target metric based on time since last scale event.",
  
    CorrectAnswer: "Reduce the cooldown period for automatic scaling.",
      Description: "When scaling responsiveness is not as fast as you would like, you should look at the cooldown period. The cooldown period is a duration when scale events will be ignored, allowing the new instances to become established and take on load. Decreasing this value will launch new variant instance faster. Automatically Scale Amazon SageMaker Models - Amazon SageMaker"
  },
  {
    Question: "Your company currently owns a fleet of restaurant supply delivery trucks that deliver fresh produce to restaurants across the city. You have developed a ML model to dynamically control the temperature of the refrigeration unit on the truck which could have huge cost savings. The model is based on XGBoost and you would like to deploy the model on each truck using a locally installed Raspberry Pi. Is this feasible given current technology?",
    OptionA: "Yes, you can deploy the model using Amazon Robomaker using the native ARM support.",
    OptionB: "No, a Raspberry Pi is not powerful enough to run an ML model using XGBoost.",
    OptionC: "Yes, you can use SageMaker Neo to compile the model into a format that is optimized for the ARM processor on the Raspberry Pi.",
    OptionD: "No, XGBoost cannot be compiled to run on an ARM processor. It can only run on x86 architectures.",
       CorrectAnswer: "Yes, you can use SageMaker Neo to compile the model into a format that is optimized for the ARM processor on the Raspberry Pi.",
   
    Description: "SageMaker Neo provides a way to compile XGBoost models which are optimized for the ARM processor in the Raspberry Pi. Amazon SageMaker Neo - Amazon SageMaker"
  },
  {
    Question: "To satisfy an external security auditor, you need to demonstrate that you can monitor all traffic going in and out of your VPC containing your deployed SageMaker model. What would you show the auditor to satisfy this audit requirement?",
    OptionA: "CloudWatch Events",
    OptionB: "SageMaker Logs",
    OptionC: "CloudWatch Alerts",
    OptionD: "VPC Flow Logs",
     CorrectAnswer: "VPC Flow Logs",
  
    Description: "VPC Flow Logs allow you to allows you to monitor all network traffic in and out of your model containers within a VPC. VPC Flow Logs - Amazon Virtual Private Cloud"
  },
  {
    Question: "You have been asked to help design a customer service bot that can help answer the most common customer service questions posed on a public chat service. Which of the following might meet the need and do so with the minimum overhead?",
    OptionA: "Amazon Lex",
    OptionB: "SageMaker Object2Vec",
    OptionC: "SageMaker BotOps",
    OptionD: "Amazon Polly",
       CorrectAnswer: "Amazon Lex",
   
    Description: "Amazon Lex can be used to create a chat bot that can understand natural language. As a service, it does not require any EC2 instances or models to be deployed before using and therefore has less overhead than a customized model using SageMaker. Amazon Lex – Build Conversation Bots"
  },
  {
    Question: "You are preparing for the deployment of an ML model based on DeepAR deployed using SageMaker Hosting Services. This model is used for real-time inferences in the financial sector. Management is concerned about a damaged reputation if the service suffers an outage. Which of the following should you do to increase fault-tolerance?",
    OptionA: "Include Elastic Inference in the endpoint configuration.",
    OptionB: "Ensure that InitialInstanceCount is at least 2 or more in the endpoint production variant.",
    OptionC: "Create a duplicate endpoint in another region using Amazon Forecast.",
    OptionD: "Recommend that they deploy using EKS in addition to the SageMaker Hosting deployment.",
      CorrectAnswer: "Ensure that InitialInstanceCount is at least 2 or more in the endpoint production variant.",
       Description: "AWS recommends that customers deploy a minimum of 2 instances for mission critical workloads when using SageMaker Hosting Services. SageMaker will automatically spread multiple instances across different AZs within a region. Deploy Multiple Instances Across Availability Zones - Amazon SageMaker"
  },
  {
    Question: "You work with a non-profit museum that would like to create audio guides that visitors can download to their mobile devices and play as they walk through the exhibits. You want to offer the audio guides in as many languages as possible but you have no budget for hiring people to create the narrations. What is a possible solution?",
    OptionA: "Use Amazon Translate to translate the script into multiple languages and then Amazon Polly to create native language audio recordings of the audio guide.",
    OptionB: "Use Amazon Textract to analyze the script and extract key themes. Use Amazon Comprehend to convert the themes into native-language narrations. Upload the files to S3 for easy downloading.",
    OptionC: "Use Amazon Polly to create a narration from the script and then Amazon Lex to dynamically adjust the audio to a variety of languages based on visitor preference.",
    OptionD: "Use Amazon Personalize to create unique narrations for each museum visitor on-demand. Personalize will automatically convert the narration to the visitor's native language.",
    CorrectAnswer: "Use Amazon Translate to translate the script into multiple languages and then Amazon Polly to create native language audio recordings of the audio guide.",
       Description: "Amazon Translate can translate text into many different languages. Once translated, Amazon Polly can then use a voice that is optimized for certain languages to create recorded narration files. Amazon Polly Amazon Translate – Neural Machine Translation - AWS"
  },
  {
    Question: "Your model consists of a linear series of steps using LDA, Random Cut Forest and a scikit-learn step. What is the most efficient way to deploy this model?",
    OptionA: "Use Data Pipeline to chain the steps together.",
    OptionB: "Use AWS Glue to chain the transform steps together.",
    OptionC: "Use AWS Step Functions to chain the steps together.",
    OptionD: "Use inference pipelines to chain the steps together.",
   
    CorrectAnswer: "Use inference pipelines to chain the steps together.",
       Description: "For this situation, the most efficient way of creating a chained series of algorithm steps is to use SageMaker Inference Pipeline. It handles transitioning the data to the next operation in the chain and deploys the containers on the same EC2 instance for efficiency. Deploy an Inference Pipeline - Amazon SageMaker"
  },
  {
    Question: "You are building out a machine learning model using multiple algorithms. You are at the point where you feel like one of the models is ready for production but you want to test difference variants of the model and compare the inference results by directing only a small amount of the traffic to the new model. What is the simplest way for you to test different model variants before allowing all traffic to go into your validated production model?",
    OptionA: "Use Amazon SageMaker to deploy the different versions of the model to a multiple endpoints. Use a Network Load Balancer to route a percentage of traffic to each model. Evaluate the results and use Route53 to route 100% of traffic to higher evaluated model.",
    OptionB: "Use Amazon SageMaker to deploy the different versions of the model to a multiple endpoints. Use a Application Load Balancer to route a percentage of traffic to each model. Evaluate the results and use Route53 to route 100% of traffic to higher evaluated model.",
    OptionC: "Use multiple EC2 instances to deploy the model on Deep Learning AMIs. Evaluate the results and reroute 100% of traffic to higher evaluated model.",
    OptionD: "Use Amazon SageMaker to deploy the different versions of the model to a single endpoint and route a percentage of traffic to each model. Evaluate the results and reroute 100% of traffic to higher evaluated model.",
    CorrectAnswer: "Use Amazon SageMaker to deploy the different versions of the model to a single endpoint and route a percentage of traffic to each model. Evaluate the results and reroute 100% of traffic to higher evaluated model.",
       Description: "You can deploy multiple variants of a model to the same Amazon SageMaker HTTPS endpoint. This is useful for testing variations of a model in production. For example, suppose that you've deployed a model into production. You want to test a variation of the model by directing a small amount of traffic, say 5%, to the new model. To do this, create an endpoint configuration that describes both variants of the model. Deploy a Model on Amazon SageMaker Hosting Services - Amazon SageMaker"
  },
  {
    Question: "Your company is preparing for a new release of a very key machine learning service that is sold to other organizations on a SaaS basis. Because company reputation is at stake, it is critical that the updates are not used in production until regression testing has shown that the updates perform as good as the existing model. Which validation strategy would you choose?",
    OptionA: "Use a rolling upgrade to determine if the model is ready for production.",
    OptionB: "Use a canary deployment to collect data on whether the model is ready for production.",
    OptionC: "Make use of backtesting with historic data.",
    OptionD: "Deploy using a Big Bang method and quickly rollback if customers report errors.",
       CorrectAnswer: "Make use of backtesting with historic data.",
       Description: "Because we must demonstrate that the updates perform as well as the existing model before we can use it in production, we would be seeking an offline validation method. Both k-fold and backtesting with historic data are offline validation methods and will allow us to evaluate the model performance without having to use live production traffic. Validate a Machine Learning Model - Amazon SageMaker"
  }
];

let currentQuestion = 0;
let correctCount = 0;
let incorrectCount = 0;

function displayQuestion() {
  const quizElement = document.getElementById('quiz');
  const questionData = quizData[currentQuestion];
  
  const html = `
    <div class="question">
      <h2>${questionData.Question}</h2>
    </div>
    <div class="options">
      <label class="option"><input type="radio" name="option" value="${questionData.OptionA}"> ${questionData.OptionA}</label>
      <label class="option"><input type="radio" name="option" value="${questionData.OptionB}"> ${questionData.OptionB}</label>
      <label class="option"><input type="radio" name="option" value="${questionData.OptionC}"> ${questionData.OptionC}</label>
      <label class="option"><input type="radio" name="option" value="${questionData.OptionD}"> ${questionData.OptionD}</label>
    </div>
    <div id="result" class="result"></div>
  `;
  
  quizElement.innerHTML = html;
}

function nextQuestion() {
  const selectedOption = document.querySelector('input[name="option"]:checked');
  if (!selectedOption) {
    alert('Please select an option');
    return;
  }
  
  const selectedValue = selectedOption.value;
  const correctAnswer = quizData[currentQuestion].CorrectAnswer;
  const description = quizData[currentQuestion].Description;
  const resultElement = document.getElementById('result');
  
  if (selectedValue === correctAnswer) {
    selectedOption.parentNode.classList.add('correct-answer');
    resultElement.innerText = "Correct!";
    correctCount++;
  } else {
    selectedOption.parentNode.classList.add('incorrect-answer');
    resultElement.innerText = "Incorrect!";
    incorrectCount++;
  }

  currentQuestion++;
  // Delay displaying the next question for 1 second
  setTimeout(displayNextQuestion, 1000);
}

function displayNextQuestion() {
  if (currentQuestion < quizData.length) {
    displayQuestion();
  } else {
    showResult();
  }
}

function submitQuiz() {
  // Calculate and display final result
  showResult();
  // Disable buttons to prevent further interaction
  document.querySelectorAll('input[type="radio"]').forEach(radio => {
    radio.disabled = true;
  });
  document.querySelectorAll('button').forEach(button => {
    button.disabled = true;
  });
}

function showResult() {
  const resultElement = document.getElementById('result');
  const html = `
    <div class="result correct-result">
      <p>Correct Answers: ${correctCount}</p>
      <p>Incorrect Answers: ${incorrectCount}</p>
    </div>
  `;
  resultElement.innerHTML = html;
}

displayQuestion();
