function startQuiz() {
  document.querySelector('.homepage-container').style.display = 'none';
  document.querySelector('.quiz-container').style.display = 'block';
}
const quizData = [
  {
    Question: "As a cloud architect, you are responsible for preparing a migration plan for your company. You want to migrate Apache Kafka (real-time streaming data pipelines) to the Google Cloud. Which Google Cloud service should you use?",
    OptionA: "Cloud Pub/Sub",
    OptionB: "Cloud Bigtable",
    OptionC: "Cloud Bigtable",
    OptionD: "Cloud Datastore",
    CorrectAnswer: "Cloud Pub/Sub",
    Discription: "Cloud Pub/Sub -> Correct. It is a fully managed real-time messaging service that enables you to send and receive messages between independent applications. It provides a scalable, durable message queue that can handle millions of messages per second. Apache Kafka is a popular messaging system used for real-time streaming data pipelines. Cloud Pub/Sub provides similar functionality as Kafka, such as message ordering and at-least-once delivery, making it a good choice for migrating Kafka workloads to the cloud."
  },
  {
    Question: "Suppose your company has two VPC networks and they want to communicate internally between these networks. So that traffic stays within Google's network and doesn't traverse the public internet. Which service should you advise?",
    OptionA: "Cloud DNS",
    OptionB: "Cloud VPN",
    OptionC: "VPC Network Peering",
    OptionD: "Hybrid Connectivity",
    CorrectAnswer: "Hybrid Connectivity",
    Discription: "VPC Network Peering -> Correct. It is the recommended option for establishing private communication between two VPC networks within Google Cloud. This service allows VPC networks to communicate with each other using private IP addresses, and the traffic stays within Google's network, which provides faster and more secure communication between the two networks."
  },
  {
    Question: "A large media company wants to implement a system for processing user-generated content, such as images and videos, and automatically categorize them based on visual content. The system should be able to process large amounts of data in real-time and scale as needed. The company wants to use the Google Cloud Platform for this implementation. Which of the following options would be the best solution for this requirement?",
    OptionA: "Use Cloud Storage to store the user-generated content, Cloud Dataflow to process the data and categorize the content.",
    OptionB: "Use Cloud Pub/Sub to ingest the user-generated content, BigQuery to store the data, and Vertex AI to categorize the content.",
    OptionC: "Use Cloud Storage to store the user-generated content, Cloud Dataproc to process the data, and Vertex AI to categorize the content.",
    OptionD: "Use Cloud Storage to store the user-generated content, Cloud Functions to process the data and categorize the content.",
    CorrectAnswer: "Use Cloud Storage to store the user-generated content, Cloud Dataproc to process the data, and Vertex AI to categorize the content.",
    Discription: "Use Cloud Storage to store the user-generated content, Cloud Dataproc to process the data, and Vertex AI to categorize the content. -> Correct. The large media company wants to implement a system that processes user-generated content, such as images and videos, and automatically categorizes them based on visual content. The system should be able to process large amounts of data in real-time and scale as needed."
  },
  {
    Question: "You are a cloud architect tasked with designing a system to store petabytes of NoSQL data for a large-scale, global e-commerce company. The system needs to support high-speed writes and reads, provide seamless scalability, and ensure near real-time consistency. The data will be heavily used for both transactional and analytical workloads. Which Google Cloud service should you choose to meet these requirements?",
    OptionA: "BigQuery",
    OptionB: "Cloud Datastore",
    OptionC: "Cloud Bigtable",
    OptionD: "Cloud Spanner",
    CorrectAnswer: "Cloud Bigtable",
    Discription: "Cloud Bigtable -> Correct. It's designed to handle massive workloads at consistent low latency, can scale to petabytes of data, and is suitable for both transactional and analytical workloads. It provides strong consistency within a single row, but eventual consistency for reading multiple rows (which can be mitigated by proper design)."
  },
  {
    Question: "You are a cloud architect for a company that has a multi-tier application running on GCP. The application includes a load balancer, several web servers, and a back-end database. As part of your security strategy, you want to allow HTTP(S) traffic only from the load balancer to the web servers. Which of the following would be the best approach?",
    OptionA: "Create two firewall rules with the same priority, one to allow traffic from the load balancer, and one to deny all other traffic.",
    OptionB: "Create a firewall rule with a low priority (high numeric value) to allow traffic from the load balancer, and a default rule to deny all other traffic.",
    OptionC: "Create a firewall rule with a high priority (low numeric value) to allow traffic from the load balancer, and a default rule to deny all other traffic.",
    OptionD: "Create a firewall rule with a high priority (low numeric value) to deny all traffic, and a rule with a low priority (high numeric value) to allow traffic from the load balancer.",
    CorrectAnswer: "Create a firewall rule with a high priority (low numeric value) to allow traffic from the load balancer, and a default rule to deny all other traffic.",
    Discription: "Create a firewall rule with a high priority (low numeric value) to allow traffic from the load balancer, and a default rule to deny all other traffic. -> Correct. Firewall rules with a lower numeric value have higher priority and are evaluated before rules with a higher numeric value. This setup ensures that the allowed traffic from the load balancer is not blocked by the default rule."
  },
  {
    Question: "As a cloud architect, you're working for a large e-commerce company that wants to detect anomalies in their sales data in real-time. The data is stored in BigQuery and is updated every few minutes with new sales information from around the globe. The goal is to quickly identify unusual spikes or drops in sales, so the business can take immediate actions. What would be the most efficient way to architect this solution?",
    OptionA: "Use Cloud Storage to store the sales data and periodically run batch processing jobs using Dataflow to detect anomalies.",
    OptionB: "Use Cloud DLP (Data Loss Prevention) to monitor the sales data in BigQuery for anomalies.",
    OptionC: "Use BigQuery ML to create a machine learning model and use scheduled queries to periodically perform anomaly detection on the recent data.",
    OptionD: "Use Cloud Monitoring to monitor the sales data in BigQuery for anomalies.",
    CorrectAnswer: "Use BigQuery ML to create a machine learning model and use scheduled queries to periodically perform anomaly detection on the recent data.",
    Discription: "Use BigQuery ML to create a machine learning model and use scheduled queries to periodically perform anomaly detection on the recent data. -> Correct. BigQuery ML enables users to create and execute machine learning models in BigQuery using SQL queries, which is perfect for the given requirement. Scheduled queries can be used to perform anomaly detection on the latest data periodically."
  },
  {
    Question: "Your customer is planning to run a big data analytics workload in Google Cloud. The customer has the following requirements: the workload must be able to handle a large volume of data and scale dynamically the workload must be able to process data in parallel the workload must be able to handle failures and recover gracefully the data must be easily queryable using SQL-like syntax Which Google Cloud service should you recommend to meet these requirements?",
    OptionA: "Cloud Dataproc",
    OptionB: "Cloud BigQuery",
    OptionC: "Cloud SQL",
    OptionD: "Cloud Dataflow",
    CorrectAnswer: "Cloud BigQuery",
    Discription: "Cloud BigQuery -> Correct. It is a fully-managed, serverless data warehouse that can handle large volumes of data and scale dynamically as required. It is designed for processing data in parallel, and can handle failures and recover gracefully. It supports querying data using SQL-like syntax, which makes it easy for users to interact with the data."
  },
  {
    Question: "An application processes a significant number of transactions that exceed the capabilities of a single virtual machine. As a cloud architect, you want to spread transactions across multiple servers in real time and in the most cost-effective way. What should you do?",
    OptionA: "You should send transactions to Pub/Sub and process them in virtual machines in a Managed Instance Group.",
    OptionB: "You should send transactions to Cloud Bigtable, and poll for new transactions from the VMs.",
    OptionC: "You should set up Cloud SQL with a memory cache for speed. On your multiple servers, poll for transactions that do not have the ‘processed’ key, and mark them ‘processed’ when done.",
    OptionD: "You should send transactions to BigQuery. On the virtual machines, poll for transactions that don't have the ‘processed’ key, and mark them ‘processed’ when done.",
    CorrectAnswer: "You should send transactions to Pub/Sub and process them in virtual machines in a Managed Instance Group.",
    Discription: "You should send transactions to Pub/Sub and process them in virtual machines in a Managed Instance Group. -> Correct. Pub/Sub is a messaging service that allows decoupling between the senders and receivers of messages, enabling the sender to send messages without worrying about who will receive them. A Managed Instance Group is a group of virtual machine instances that are managed together and can be scaled up or down automatically based on the workload. By using Pub/Sub and a Managed Instance Group, the transactions can be sent to Pub/Sub, and the virtual machines can subscribe to the relevant Pub/Sub topic to receive and process the transactions. This approach ensures that the transactions are spread across multiple servers, allowing for parallel processing and improved performance."
  },
  {
    Question: "Your customer is planning to deploy a web application that requires high performance and low latency. The customer has the following requirements: the application must be easily deployable and manageable the application must be highly available and recover from failures automatically the application must be able to handle incoming traffic spikes and scale dynamically the application must be cost-effective Which Google Cloud service should you recommend to meet these requirements?",
    OptionA: "App Engine",
    OptionB: "Cloud Load Balancer",
    OptionC: "Google Kubernetes Engine",
    OptionD: "Cloud Functions",
    CorrectAnswer: "App Engine",
    Discription: "App Engine -> Correct. The customer's requirements suggest that they need a serverless platform that can scale dynamically, be highly available, and cost-effective. App Engine meets all these requirements as it is a fully managed platform that can automatically scale up or down based on incoming traffic and provides high availability and fault tolerance. Additionally, it offers a simple and easy way to deploy and manage web applications."
  },
  {
    Question: "In your company, each employee has a credit card assigned to their account. As a cloud architect, you want to consolidate the billing of all GCP projects into a new billing account. With Google best practices in mind, how should you do it?",
    OptionA: "In the GCP Console, move all projects to the root organization in the Resource Manager.",
    OptionB: "You should send an email to Google Billing Support and request them to create a new billing account and link all the projects to the billing account.",
    OptionC: "You should create a new Billing account and set up a payment method with company credit card.",
    OptionD: "Once a credit card is assigned to the project, it cannot be changed. You have to create all the projects from scratch with new billing account.",
    CorrectAnswer: "In the GCP Console, move all projects to the root organization in the Resource Manager.",
    Discription: "In the GCP Console, move all projects to the root organization in the Resource Manager. -> Correct. Moving all projects to the root organization in the Resource Manager allows you to create a new billing account at the organization level and link all the projects to it. This approach is recommended by Google as it helps to centralize billing and provides better visibility into the costs and usage of all GCP projects. Once the projects are linked to the new billing account, you can remove the credit cards associated with individual accounts and use a single payment method for all projects."
  },
  {
    Question: "You're a cloud architect and have been assigned a task to develop a model that will predict the type of product a customer is most likely to purchase next, based on their past purchases and behavior. The client has a massive amount of historic data available but lacks machine learning expertise. Furthermore, the solution needs to be able to constantly learn from new data. Which of the following would be the best approach for this situation?",
    OptionA: "Use BigQuery ML, retrain the model manually with new data.",
    OptionB: "Use Cloud Machine Learning Engine with TensorFlow and retrain the model manually with new data.",
    OptionC: "Use AutoML Tables, retrain the model periodically with new data.",
    OptionD: "Use AutoML Vision, retrain the model periodically with new data.",
    CorrectAnswer: "Use AutoML Tables, retrain the model periodically with new data.",
    Discription: "Use AutoML Tables, retrain the model periodically with new data. -> Correct. AutoML Tables is the right solution because it is specifically designed for tabular or structured data, like historic purchase data. AutoML allows for automatic model training, tuning, and deployment, which suits the lack of machine learning expertise. It also allows the model to be retrained periodically with new data, which meets the requirement of constantly learning from new data."
  },
  {
    Question: "Your customer is planning to store and analyze large amounts of structured and unstructured data in Google Cloud. The customer has the following requirements:the data must be stored in a highly available and scalable storage system the data must be easily searchable and filterable the data must be easily queryable using SQL-like syntax the data must be easily integratable with other Google Cloud services Which Google Cloud service should you recommend to meet these requirements?",
    OptionA: "Cloud Datastore",
    OptionB: "Cloud Storage",
    OptionC: "Cloud Firestore",
    OptionD: "Cloud Bigtable",
    CorrectAnswer: "Cloud Firestore",
    Discription: "Cloud Firestore -> Correct. Cloud Firestore is a fully managed NoSQL document database that is designed to store and manage structured and semi-structured data. It provides a highly available and scalable storage system with automatic sharding and load balancing, making it a good choice for storing large amounts of data. Firestore also provides powerful querying capabilities, with support for SQL-like syntax and complex queries on both structured and unstructured data. It also provides a flexible data model, making it easy to store and retrieve data in a variety of formats. In addition, Firestore is easily integratable with other Google Cloud services such as App Engine, Cloud Functions, and BigQuery, making it a good choice for integrating with other services as required by the customer."
  },
  {
    Question: "A multinational retail company has stores in multiple countries and wants to leverage GCP to manage its inventory and ordering processes. The company wants to implement a mobile application that provides real-time visibility into inventory levels and enables employees to place orders for out-of-stock items from any location. The solution should also be scalable and be able to handle high volume transactions. Which of the following options would be the most effective approach to meet these requirements?",
    OptionA: "Use Cloud Storage to store the inventory data and Cloud Functions to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data.",
    OptionB: "Use Cloud BigQuery to store the inventory data and Cloud Functions to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data.",
    OptionC: "Use Cloud SQL to store the inventory data and App Engine to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data.",
    OptionD: "Use Cloud Bigtable to store the inventory data and Cloud Dataflow to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data.",
    CorrectAnswer: "Use Cloud SQL to store the inventory data and App Engine to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data.",
    Discription: "Use Cloud SQL to store the inventory data and App Engine to trigger the ordering process. Use Cloud Pub/Sub to notify employees of inventory changes and Cloud IAM to control access to the data. -> Correct. Cloud SQL is a fully-managed database service that provides a highly scalable and available relational database engine for storing and managing data. It is a good option for storing inventory data because it supports high transaction rates and can handle large volumes of data. App Engine, on the other hand, is a serverless platform for building and deploying web and mobile applications, which can be used to trigger the ordering process in response to requests from the mobile application. This makes App Engine a good choice for implementing the ordering process because it can handle high volumes of transactions and scale automatically."
  },
  {
    Question: "Your organization uses BigQuery extensively for data analysis and you are developing a new solution to automate some recurring tasks. As part of this solution, you need to create a new table in BigQuery and load data into it from a CSV file in Cloud Storage, all from a Compute Engine instance. What would be the correct approach?",
    OptionA: "Use the gsutil command to create the table in BigQuery and then the bq command-line tool to load the data.",
    OptionB: "Use the bq command-line tool to create the table and then the gsutil cp command to copy the data from Cloud Storage to BigQuery.",
    OptionC: "Use the bq command-line tool to both create the table and load the data from Cloud Storage.",
    OptionD: "Use the gcloud command-line tool to create the table and load the data into BigQuery.",
    CorrectAnswer: "Use the bq command-line tool to both create the table and load the data from Cloud Storage.",
    Discription: "Use the bq command-line tool to both create the table and load the data from Cloud Storage. -> Correct. The bq tool can be used to perform a variety of BigQuery operations, including creating tables and loading data from Cloud Storage."
  },
  {
    Question: "You're designing a CI/CD pipeline for an application hosted on GCP. As part of the pipeline, you need to programmatically deploy and manage several GCP resources. Which approach should you use?",
    OptionA: "Cloud Shell",
    OptionB: "Google Cloud Console",
    OptionC: "Google Cloud SDK and REST APIs",
    OptionD: "Cloud Deployment Manager",
    CorrectAnswer: "Google Cloud SDK and REST APIs",
    Discription: "Google Cloud SDK and REST APIs -> Correct. The Google Cloud SDK and GCP REST APIs provide the programmatic interfaces needed to manage GCP resources within a CI/CD pipeline. They allow for the automation of resource management tasks."
  },
  {
    Question: "Your organization provides a video streaming service to global users. Recently, users are complaining about the high latency when watching videos. As a cloud architect, how would you improve the streaming experience?",
    OptionA: "Use Cloud Pub/Sub to push video content to users.",
    OptionB: "Store all video content in a single region in Cloud Storage.",
    OptionC: "Use Cloud Spanner to store and serve video content.",
    OptionD: "Use Cloud CDN to cache and serve video content.",
    CorrectAnswer: "Use Cloud CDN to cache and serve video content.",
    Discription: "Use Cloud CDN to cache and serve video content. -> Correct. Cloud CDN leverages Google's global edge network to cache and serve content closer to users, which can significantly reduce latency and improve the streaming experience."
  },
  {
    Question: "Suppose you run a small business and need to grant a role for your accountant to view billing reports and approve invoices. With Google's best practices in mind, which Billing IAM role should you grant to your accountant?",
    OptionA: "Billing Account Viewer",
    OptionB: "Billing Account User Project Creator",
    OptionC: "Billing Account Creator",
    OptionD: "Billing Account Administrator",
    CorrectAnswer: "Billing Account Viewer",
    Discription: "Billing Account Viewer -. Correct. The Billing Account Viewer role allows the user to view billing reports, cost trends, and related information without being able to make any changes or perform any actions. This is the least privileged role that grants access to billing information."
  },
  {
    Question: "As a cloud architect, while utilizing the Google Network Intelligence Center's Firewall Insights feature, you observe that there are no log rows available for viewing when accessing the Firewall Insights page in the Google Cloud console. This prompts the need for assessing the effectiveness of the applied firewall ruleset, considering the existence of multiple firewall rules associated with the Compute Engine instance. To troubleshoot the issue, what steps should you take?",
    OptionA: "Verify that your user account is assigned the compute. Network Admin Identity and Access Management (IAM) role.",
    OptionB: "Enable Virtual Private Cloud (VPC) flow logging.",
    OptionC: "Install the Google Cloud SDK, and verify that there are no Firewall logs in the command line output.",
    OptionD: "Enable Firewall Rules Logging for the firewall rules you want to monitor.",
    CorrectAnswer: "Enable Firewall Rules Logging for the firewall rules you want to monitor.",
    Discription: "Enable Firewall Rules Logging for the firewall rules you want to monitor. -> Correct. Enabling Firewall Rules Logging allows you to capture logs for the specified firewall rules. By enabling logging for the relevant firewall rules, you can troubleshoot the issue of missing log rows in Firewall Insights. This step ensures that the necessary logs are generated and available for analysis, providing visibility into the effectiveness of the firewall ruleset."
  },
  {
    Question: "You plan to migrate your on-premises MySQL and PostgreSQL databases to the Google Cloud using Lift and Shift approach. Which Google Cloud service should you use?",
    OptionA: "BigQuery Data Transfer Service",
    OptionB: "Storage Transfer Service",
    OptionC: "Migrate for Anthos",
    OptionD: "Database Migration Service",
    CorrectAnswer: "Database Migration Service",
    Discription: "Database Migration Service -> Correct. Google's Database Migration Service is specifically designed to migrate databases to Google Cloud, including MySQL and PostgreSQL. It supports various migration source types, including on-premises databases, and provides a simple and automated way to migrate databases with minimal downtime."
  },
  {
    Question: "As a cloud architect, you are tasked with creating a signed URL for a Google Cloud Storage (GCS) bucket. The requirement is to allow access to a certain object in the bucket for a third-party service, but only for 30 minutes. The solution should ensure minimum privileges and security. Which of the following options should you choose to implement this?",
    OptionA: "Create a signed URL using a user-managed service account with full access on the bucket and a 30-minute expiration.",
    OptionB: "Create a signed URL using the bucket's default service account with a 30-minute expiration.",
    OptionC: "Create a signed URL using a user-managed service account with read access on the bucket and a 30-minute expiration.",
    OptionD: "Create a signed URL using the bucket's default service account with no expiration, and manually invalidate the URL after 30 minutes.",
    CorrectAnswer: "Create a signed URL using a user-managed service account with read access on the bucket and a 30-minute expiration.",
    Discription: "Create a signed URL using a user-managed service account with read access on the bucket and a 30-minute expiration. -> Correct. Creating a signed URL using a user-managed service account that only has read access to the bucket follows the principle of least privilege. The 30-minute expiration limits the availability of the object, providing further security."
  },
  {
    Question: "As a cloud architect, you are migrating an application to GCP using overnight batch jobs that take approximately one hour. Which service should you recommend to do this with minimal cost?",
    OptionA: "You should run the batch jobs in a GKE cluster.",
    OptionB: "You should run the batch jobs in a virtual machine instance with GPUs.",
    OptionC: "You should run the batch jobs in a preemptible compute engine instance.",
    OptionD: "You should run the batch jobs in a normal virtual machine instance.",
    CorrectAnswer: "You should run the batch jobs in a preemptible compute engine instance.",
    Discription: "You should run the batch jobs in a preemptible compute engine instance. -> Correct. Preemptible Compute Engine instances are a cost-effective option for running batch jobs. Preemptible instances are regular instances that are priced lower than standard instances and can run for up to 24 hours. However, these instances can be terminated at any time if Google needs to reclaim the resources, and their availability is not guaranteed. Since the batch jobs are running overnight, and the workloads are not mission-critical, it is acceptable to use preemptible instances. Additionally, the batch jobs are not GPU-intensive, so there is no need to use GPU-enabled instances."
  },
  {
    Question: "As a cloud architect, you are working on a complex scenario where you have to deploy a monitoring pod in a DaemonSet object across a GKE cluster for a client's application. You want the monitoring pod to be deployed on every node of the GKE cluster. Which approach will allow you to efficiently achieve this objective?",
    OptionA: "Deploy the monitoring pod as a ReplicaSet across the nodes in the GKE cluster.",
    OptionB: "Deploy the monitoring pod using a DaemonSet across the nodes in the GKE cluster.",
    OptionC: "Deploy the monitoring pod using a StatefulSet across the nodes in the GKE cluster.",
    OptionD: "Deploy the monitoring pod individually on each node in the GKE cluster.",
    CorrectAnswer: "Deploy the monitoring pod using a DaemonSet across the nodes in the GKE cluster.",
    Discription: "Deploy the monitoring pod using a DaemonSet across the nodes in the GKE cluster. -> Correct. A DaemonSet ensures that all (or some) nodes run a copy of a pod. This is suitable for deploying system daemons such as log collectors, monitoring services etc. When a node is added to the cluster, the pod gets added to the new node, making it ideal for our scenario."
  },
  {
    Question: "A financial services company is looking to implement a new fraud detection system for their credit card transactions. The system should be able to detect fraud in real-time and provide alerts to security personnel. The company wants to leverage the Google Cloud Platform for this implementation. Which of the following options would be the best solution for this requirement?",
    OptionA: "Use Cloud Datastore to store transaction data and use Cloud Functions to process transactions in real-time and trigger fraud alerts.",
    OptionB: "Use BigQuery to store transaction data and use Cloud Dataflow to process transactions in real-time and trigger fraud alerts.",
    OptionC: "Use Cloud SQL to store transaction data and use Cloud Pub/Sub to process transactions in real-time and trigger fraud alerts.",
    OptionD: "Use Cloud Firestore to store transaction data and use Cloud Tasks to process transactions in real-time and trigger fraud alerts.",
    CorrectAnswer: "Use BigQuery to store transaction data and use Cloud Dataflow to process transactions in real-time and trigger fraud alerts.",
    Discription: "Use BigQuery to store transaction data and use Cloud Dataflow to process transactions in real-time and trigger fraud alerts.  -> Correct. BigQuery is a powerful and scalable data warehousing solution that can handle large volumes of transaction data efficiently. Cloud Dataflow, a serverless data processing service, can be used to process transactions in real-time and trigger fraud alerts based on specific criteria. This combination of BigQuery and Cloud Dataflow provides a robust and scalable solution for real-time fraud detection in the given scenario."
  },
  {
    Question: "You are operating in a high-security environment where Compute Engine VMs are not permitted to access the public internet. Currently, you lack a VPN connection to access a local network file server. You have a necessity to deploy a certain software on a Compute Engine instance. What is the recommended method for installing the software?",
    OptionA: "Upload the necessary installation files to Cloud Storage, and implement firewall rules to block all traffic except the IP address range specific to Cloud Storage. Utilize gsutil to download the files to the VM.",
    OptionB: "Upload the necessary installation files to Cloud Storage. Configure the VM on a subnet with Private Google Access. Ensure the VM is assigned only an internal IP address. Use gsutil to download the installation files to the VM.",
    OptionC: "Upload the necessary installation files to Cloud Source Repositories, and establish firewall rules to block all traffic except the IP address range for Cloud Source Repositories. Use gsutil to download the files to the VM.",
    OptionD: "Upload the necessary installation files to Cloud Source Repositories. Set up the VM on a subnet with Private Google Access. Assign only an internal IP address to the VM. Use gcloud to download the installation files to the VM.",
    CorrectAnswer: "Upload the necessary installation files to Cloud Storage. Configure the VM on a subnet with Private Google Access. Ensure the VM is assigned only an internal IP address. Use gsutil to download the installation files to the VM.",
    Discription: "Upload the necessary installation files to Cloud Storage. Configure the VM on a subnet with Private Google Access. Ensure the VM is assigned only an internal IP address. Use gsutil to download the installation files to the VM. -> Correct. It provides a suitable solution for installing the software in a high-security environment where Compute Engine VMs are not allowed to access the public internet. In this scenario, uploading the necessary installation files to Cloud Storage allows you to store the files securely in Google Cloud. By configuring the VM on a subnet with Private Google Access, you ensure that the VM can access Google APIs and services like Cloud Storage without requiring access to the public internet. Assigning only an internal IP address to the VM further enforces the restriction on accessing the internet. Using gsutil, which is a command-line tool for interacting with Cloud Storage, you can download the installation files directly to the VM from Cloud Storage. This method allows you to retrieve the necessary files securely and efficiently within the restricted environment."
  },
  {
    Question: "A financial services company is planning to implement a high-performance computing solution on GCP to support its algorithmic trading operations. The company wants to ensure that the solution is scalable, low-latency, and able to handle a large volume of data. The company also wants to ensure that the solution is secure and that sensitive financial data is protected. Which of the following options would be the most effective approach to meet these requirements?",
    OptionA: "Use Cloud Functions to stream the data and Cloud Dataflow to process the data. Use BigQuery to store the data and Cloud IAM to control access to the data. Implement security using Cloud KMS.",
    OptionB: "Use Cloud Pub/Sub to stream the data and Cloud Dataproc to process the data. Use BigQuery to store the data and Cloud IAM to control access to the data. Implement security using Cloud IAP.",
    OptionC: "Use Gogle Cloud Pub/Sub to stream the data and Cloud Dataflow to process the data. Use Cloud Bigtable to store the data and Cloud IAM to control access to the data. Implement security using Cloud KMS.",
    OptionD: "Use Cloud Pub/Sub to stream the data and Cloud Dataproc to process the data. Use Cloud Bigtable to store the data and Cloud IAM to control access to the data. Implement security using Cloud IAP.",
    CorrectAnswer: "Use Gogle Cloud Pub/Sub to stream the data and Cloud Dataflow to process the data. Use Cloud Bigtable to store the data and Cloud IAM to control access to the data. Implement security using Cloud KMS.",
    Discription: "Use Gogle Cloud Pub/Sub to stream the data and Cloud Dataflow to process the data. Use Cloud Bigtable to store the data and Cloud IAM to control access to the data. Implement security using Cloud KMS. -> Correct. Google Cloud Pub/Sub is designed for scalable, low-latency data streaming, while Cloud Dataflow offers powerful data processing capabilities. Cloud Bigtable is a high-performance NoSQL database that can handle large volumes of data with low latency. Cloud IAM provides access control and authorization mechanisms, ensuring the security of the sensitive financial data. Cloud KMS (Key Management Service) can be used to secure data encryption keys and provide additional security measures."
  },
  {
    Question: "As a cloud architect, you are working for a company that needs to perform a large-scale data processing task that is highly parallelizable and not time-sensitive. This operation should be completed within a reasonable time frame, but it doesn't require an immediate result. The cost of operation is a major consideration. The company has asked you for the most cost-effective solution that matches these requirements. What would be your recommendation?",
    OptionA: "Use Google Cloud Functions for the data processing task.",
    OptionB: "Use Google Cloud Dataflow with non-preemptible Compute Engine machines.",
    OptionC: "Use Google Cloud Dataproc with preemptible Compute Engine machines.",
    OptionD: "Use Google Kubernetes Engine with preemptible Compute Engine machines.",
    CorrectAnswer: "Use Google Cloud Dataproc with preemptible Compute Engine machines.",
    Discription: "Use Google Cloud Dataproc with preemptible Compute Engine machines. -> Correct. Dataproc is designed for fast, easy, and cost-efficient processing of big data workloads. Using preemptible Compute Engine machines, which are significantly cheaper than standard machines, can save costs. Since the task is not time-sensitive, the fact that preemptible instances can be terminated at any time (up to a maximum of 24 hours) is not a significant issue."
  },
  {
    Question: "Mountkirk Games aims to establish a continuous delivery pipeline for their architecture, which comprises numerous small services requiring quick updates and rollbacks. The following requirements are specified by Mountkirk Games: redundant deployment of services across multiple regions in the US and Europe. only the frontend services are accessible via the public internet allocation of a single frontend IP address for their entire service fleet deployment artifacts are immutable Which product combination would be most suitable for their needs?",
    OptionA: "Cloud Storage, Cloud Dataflow, Compute Engine",
    OptionB: "Cloud Functions, Cloud Pub/Sub, Cloud Deployment Manager",
    OptionC: "Cloud Storage, App Engine, Cloud Load Balancing",
    OptionD: "Container Registry, Google Kubernetes Engine, Cloud Load Balancing",
    CorrectAnswer: "Container Registry, Google Kubernetes Engine, Cloud Load Balancing",
    Discription: "Container Registry, Google Kubernetes Engine, Cloud Load Balancing -> Correct. Google Kubernetes Engine is ideal for deploying small services that can be updated and rolled back quickly. It is a best practice to manage services using immutable containers. Cloud Load Balancing supports globally distributed services across multiple regions. It provides a single global IP address that can be used in DNS records. Using URL Maps, the requests can be routed to only the services that Mountkirk wants to expose. Container Registry is a single place for a team to manage Docker images for the services."
  },
  {
    Question: "When handling Personally Identifiable Information (PII) in Google Cloud Platform (GCP), which of the following statements regarding Customer-managed encryption keys (CMEK) is correct?",
    OptionA: "CMEK ensures compliance with data protection regulations, eliminating the need for customers to implement additional security measures.",
    OptionB: "CMEK provides customers with full control and ownership of encryption keys, allowing them to manage the encryption of all types of data within GCP.",
    OptionC: "CMEK is a feature exclusively available for Google-managed services and cannot be utilized for customer-owned resources.",
    OptionD: "CMEK is automatically enabled by default for all GCP services, providing enhanced encryption and protection for PII.",
    CorrectAnswer: "CMEK provides customers with full control and ownership of encryption keys, allowing them to manage the encryption of all types of data within GCP.",
    Discription: "CMEK provides customers with full control and ownership of encryption keys, allowing them to manage the encryption of all types of data within GCP. -> Correct. CMEK provides customers with full control and ownership of encryption keys, allowing them to manage the encryption of all types of data within GCP. This statement accurately represents the purpose and functionality of CMEK, empowering customers to maintain control over their encryption keys and data security."
  },
  {
    Question: "Refer to the Mountkirk Games case study for this question: https://services.google.com/fh/files/blogs/master_case_study_mountkirk_games.pdf As your new game is currently in public beta on the Google Cloud platform, it is essential to establish significant service level objectives (SLOs) before its official release to the public. What steps should you take to accomplish this?",
    OptionA: "You should define one SLO as total uptime of the game server within a week. Define the other SLO as the mean response time of all HTTP requests that are less than 100 ms.",
    OptionB: "You should define one SLO as 99.9% game server availability. Define the other SLO as less than 100-ms latency.",
    OptionC: "You should define one SLO as service availability that is the same as Google Cloud's availability. Define the other SLO as 100-ms latency.",
    OptionD: "You should define one SLO as 99% HTTP requests return the 2xx status code. Define the other SLO as 99% requests return within 100 ms.",
    CorrectAnswer: "You should define one SLO as 99% HTTP requests return the 2xx status code. Define the other SLO as 99% requests return within 100 ms.",
    Discription: "You should define one SLO as 99% HTTP requests return the 2xx status code. Define the other SLO as 99% requests return within 100 ms. -> Correct. It clearly defines the service level indicators and how to measure them."
  },
  {
    Question: "In your role as a cloud architect, your task involves deploying PHP App Engine Standard alongside Cloud SQL as the backend, with the goal of reducing the number of queries to the database. What actions should you take?",
    OptionA: "Set the memcache service level to shared. Create a cron task that runs every minute to save all expected queries to a key called cached_queries.",
    OptionB: "Set the memcache service level to shared. Create a key called cached_queries, and return database values from the key before using a query to Cloud SQL.",
    OptionC: "Set the memcache service level to dedicated. Create a cron task that runs every minute to populate the cache with keys containing query results.",
    OptionD: "Set the memcache service level to dedicated. Create a key from the hash of the query, and return database values from memcache before issuing a query to Cloud SQL.",
    CorrectAnswer: "Set the memcache service level to dedicated. Create a key from the hash of the query, and return database values from memcache before issuing a query to Cloud SQL.",
    Discription: "Set the memcache service level to dedicated. Create a key from the hash of the query, and return database values from memcache before issuing a query to Cloud SQL. -> Correct. By setting the memcache service level to dedicated, you ensure that you have a dedicated cache for your application, which can provide better performance and reliability. Creating a key from the hash of the query allows you to uniquely identify the result of each query. By checking if the result exists in memcache before issuing a query to Cloud SQL, you can reduce the number of queries to the database and retrieve the data directly from the cache if it's available."
  },
  {
    Question: "Your organization uses a BigQuery data warehouse for analytics. You expect the volume of data and the complexity of queries to increase significantly over the next year. What could be a strategy to ensure query performance remains high?",
    OptionA: "Migrate data from BigQuery to Cloud Spanner for increased performance.",
    OptionB: "Increase the amount of storage available to BigQuery.",
    OptionC: "Use the BigQuery Reservation model to purchase dedicated query processing capacity.",
    OptionD: "Split your data into multiple BigQuery datasets to balance the load.",
    CorrectAnswer: "Use the BigQuery Reservation model to purchase dedicated query processing capacity.",
    Discription: "Use the BigQuery Reservation model to purchase dedicated query processing capacity. -> Correct. The BigQuery Reservation model allows you to purchase dedicated query processing capacity (slots). This can ensure consistent performance even as query load increases."
  },
  {
    Question: "A large logistics company is looking to build a secure and scalable platform for tracking shipments and managing delivery schedules. The platform must meet the following requirements: support real-time tracking of shipments with high accuracy ensure secure storage and processing of sensitive shipment information enable efficient coordination of delivery schedules across multiple locations minimize downtime during maintenance and upgrades minimize costs while still providing high performance Which solution would you recommend to meet these requirements?",
    OptionA: "Implementing a custom-built solution using Cloud Pub/Sub for real-time data processing, Bigtable for data storage, and Google Kubernetes Engine (GKE) for deployment and scaling.",
    OptionB: "Implementing a managed solution using Google Maps Platform for real-time tracking, Cloud Datastore for data storage, and Cloud Identity and Access Management (IAM) for secure access control.",
    OptionC: "Implementing a hybrid solution using Cloud SQL for data storage, Compute Engine for data processing, and BigQuery for real-time analytics.",
    OptionD: "Implementing a serverless solution using Cloud Functions for data processing, Cloud Firestore for data storage, and Cloud Pub/Sub for real-time data processing.",
    CorrectAnswer: "Implementing a managed solution using Google Maps Platform for real-time tracking, Cloud Datastore for data storage, and Cloud Identity and Access Management (IAM) for secure access control.",
    Discription: "Implementing a managed solution using Google Maps Platform for real-time tracking, Cloud Datastore for data storage, and Cloud Identity and Access Management (IAM) for secure access control. -> Correct. Google Maps Platform provides real-time location tracking capabilities that can be used to track shipments. Cloud Datastore is a highly scalable, managed NoSQL database that can be used to store sensitive shipment information securely. And, Cloud Identity and Access Management (IAM) can be used to control access to the platform, ensuring that sensitive information is only accessible to authorized individuals. This solution would meet all the requirements while still providing high performance and minimizing costs."
  },
  {
    Question: "You are a cloud architect and have been tasked with creating a data retention policy on a Google Cloud Storage (GCS) bucket that holds sensitive client information. This policy should ensure that objects older than 90 days are not accessible, even if they haven't been deleted. Also, to ensure business continuity, deleted objects should be restorable within 5 days. Which of the following methods would be the most suitable for implementing this policy?",
    OptionA: "Apply a 90-day lifecycle rule to archive objects and enable versioning with a 5-day lifecycle rule to delete previous versions of objects.",
    OptionB: "Apply a 90-day retention policy to the bucket and a 5-day lifecycle rule to delete older versions of objects.",
    OptionC: "Apply a 90-day lifecycle rule to delete objects, enable versioning, and use a 5-day lifecycle rule to delete previous versions of objects.",
    OptionD: "Apply a 90-day lifecycle rule to delete objects and a 5-day retention policy to retain deleted objects.",
    CorrectAnswer: "Apply a 90-day lifecycle rule to delete objects, enable versioning, and use a 5-day lifecycle rule to delete previous versions of objects.",
    Discription: "Apply a 90-day lifecycle rule to delete objects, enable versioning, and use a 5-day lifecycle rule to delete previous versions of objects. -> Correct. A lifecycle rule with a 90-day condition will ensure objects older than 90 days are deleted. Bucket versioning will keep all versions of an object, including those that have been deleted. A second lifecycle rule with a 5-day condition on previous versions will ensure that deleted objects can be restored within 5 days."
  },
  {
    Question: "Your organization has an existing system on-premise and plans to migrate to Google Cloud. However, they plan to keep some parts of their system on-premises for the foreseeable future. The goal is to have the on-premise and cloud systems communicate securely and efficiently. What should be your strategy?",
    OptionA: "Use VPC Network Peering to connect the on-premise system to the VPC in Google Cloud",
    OptionB: "Use Cloud Interconnect to connect the on-premise system to Google Cloud",
    OptionC: "Use Cloud VPN to establish a secure connection between the on-premise system and Google Cloud",
    OptionD: "Use Cloud Endpoints to create APIs for the on-premise system and access them from Google Cloud",
    CorrectAnswer: "Use Cloud VPN to establish a secure connection between the on-premise system and Google Cloud",
    Discription: "Use Cloud VPN to establish a secure connection between the on-premise system and Google Cloud -> Correct. Cloud VPN can create a secure connection between an on-premise system and Google Cloud over the internet, which is ideal for hybrid systems that span on-premise and cloud environments."
  },
  {
    Question: "Your company runs several critical applications on Google Cloud. There has been a significant increase in the number of user-reported incidents recently, indicating performance issues. As a cloud architect, how would you improve the monitoring to proactively identify and address performance issues?",
    OptionA: "Use Cloud Trace to trace all requests to the applications.",
    OptionB: "Use Cloud Profiler to continuously profile the applications.",
    OptionC: "Use Cloud Debugger to debug the applications in real-time.",
    OptionD: "Set up custom metrics in Cloud Monitoring for all critical applications and configure alerting based on those metrics.",
    CorrectAnswer: "Set up custom metrics in Cloud Monitoring for all critical applications and configure alerting based on those metrics.",
    Discription: "Set up custom metrics in Cloud Monitoring for all critical applications and configure alerting based on those metrics. -> Correct. Cloud Monitoring allows you to set up custom metrics for monitoring specific aspects of your applications, and you can configure alerts to be notified when these metrics cross certain thresholds. This proactive approach can help you address performance issues before they impact users."
  },
  {
    Question: "You need to migrate your legacy on-premises applications to Google Cloud that are written in C++ and you want to use the serverless approach. What GCP compute services should you use?",
    OptionA: "You should deploy this application using a Managed Instance Group.",
    OptionB: "You should deploy the containerized version of the application in App Engine Flexible.",
    OptionC: "You should deploy the containerized version of the application in Cloud Run.",
    OptionD: "You should deploy the containerized version of the application in Google Kubernetes Engine.",
    CorrectAnswer: "You should deploy the containerized version of the application in Cloud Run.",
    Discription: "You should deploy the containerized version of the application in Cloud Run. -> Correct. Cloud Run is a serverless compute platform that allows you to run stateless containers in a fully managed environment. It supports containerized applications built on any language or framework, including C++. By deploying the containerized version of the application to Cloud Run, you can take advantage of the serverless approach, where you only pay for the actual usage of resources, and the environment is fully managed by Google Cloud."
  },
  {
    Question: "As a cloud architect, you are assisting a global organization in migrating its petabyte-scale on-premises data to Google Cloud. The organization has an extremely slow internet connection, strict compliance and security standards, and cannot tolerate any downtime during working hours. The migration should be completed as quickly as possible without interrupting business operations. Which of the following strategies would you recommend?",
    OptionA: "Use Storage Transfer Service for online transfer.",
    OptionB: "Use Transfer Appliance to ship data to Google Cloud.",
    OptionC: "Use Cloud Dataflow to process and migrate data.",
    OptionD: "Use gsutil command-line tool to upload data to Cloud Storage.",
    CorrectAnswer: "Use Transfer Appliance to ship data to Google Cloud.",
    Discription: "Use Transfer Appliance to ship data to Google Cloud. -> Correct. This hardware appliance can be loaded with data and physically shipped to Google, where the data can be uploaded to the cloud. This approach is particularly useful in scenarios where an organization has a slow or unrelia."
  },
  {
    Question: "Your organization operates a multi-tier web application on Google Cloud. You have been tasked with implementing a solution to ensure operational reliability and quickly identify any issues that might affect the application's performance. Which of the following would be the best approach?",
    OptionA: "Enable Cloud Debugger on all application instances.",
    OptionB: "Enable Cloud Logging and Monitoring for the entire project.",
    OptionC: "Use Cloud Dataflow to process logs in real-time.",
    OptionD: "Configure Cloud Pub/Sub to publish messages whenever there is a change in the application.",
    CorrectAnswer: "Enable Cloud Logging and Monitoring for the entire project.",
    Discription: "Enable Cloud Logging and Monitoring for the entire project. -> Correct. This option includes both Logging and Monitoring, which can be used to collect logs, metrics, events, and metadata from your Cloud project. This will allow you to monitor the performance of your application and quickly identify any issues."
  },
  {
    Question: "Your company has an existing monolithic application running on-premises and you've been tasked to migrate this application to Google Cloud Platform (GCP). The company wants to start taking advantage of microservices for scalability and maintainability. What strategy should you use?",
    OptionA: "Refactor the monolithic application into microservices and deploy each one as a Cloud Function.",
    OptionB: "Migrate the monolithic application to App Engine Standard, then refactor for microservices.",
    OptionC: "Refactor the monolithic application into microservices and deploy using GKE.",
    OptionD: "Migrate the application to Cloud Run without refactoring, then move individual services to GKE as they are broken out.",
    CorrectAnswer: "Refactor the monolithic application into microservices and deploy using GKE.",
    Discription: "Refactor the monolithic application into microservices and deploy using GKE. -> Correct. This is a good strategy because Google Kubernetes Engine (GKE) is designed to manage, scale, and deploy containerized applications, which are well-suited to a microservices architecture."
  },
  {
    Question: "You are a cloud architect tasked with designing an application to regularly fetch and process social media data every 15 minutes. The processed data is then stored in Cloud Storage and made available to an analytics team. This operation must be reliable and scalable to handle periods of high demand, while minimizing costs. Which of the following is the most suitable approach to handle this scenario?",
    OptionA: "Use App Engine Flexible with the Cron service to run a dedicated service for fetching and processing the data every 15 minutes.",
    OptionB: "Use App Engine Standard with the Cron service to trigger a Cloud Run service every 15 minutes. The service retrieves and processes the data and then stores it in Cloud Storage.",
    OptionC: "Use Compute Engine with a Cron job installed on the instance to fetch, process, and store the data.",
    OptionD: "Use Cloud Scheduler to trigger a Pub/Sub topic every 15 minutes, which then triggers a Dataflow job to fetch, process, and store the data.",
    CorrectAnswer: "Use App Engine Standard with the Cron service to trigger a Cloud Run service every 15 minutes. The service retrieves and processes the data and then stores it in Cloud Storage.",
    Discription: "Use App Engine Standard with the Cron service to trigger a Cloud Run service every 15 minutes. The service retrieves and processes the data and then stores it in Cloud Storage. -> Correct. App Engine's Cron service can reliably trigger tasks on a schedule. Cloud Run can handle processing tasks and automatically manages and scales the underlying infrastructure, allowing it to handle high-demand periods. Cloud Run services only run when they're needed, which can help minimize costs."
  },
  {
    Question: "You are designing a cloud-based architecture for a company that requires a secure and scalable solution for its database. The database must be able to handle high volumes of transactions, and the data must be encrypted at rest and in transit. Which of the following options provides the best solution?",
    OptionA: "Use a managed database service with automatic encryption and scale-up capability.",
    OptionB: "Use a NoSQL database service with built-in encryption and automatic scaling.",
    OptionC: "Use a self-managed database service with manual encryption and scale-out capability.",
    OptionD: "Use a hybrid approach with a self-managed database service and a third-party encryption tool.",
    CorrectAnswer: "Use a managed database service with automatic encryption and scale-up capability.",
    Discription: "Use a managed database service with automatic encryption and scale-up capability. -> Correct. Using a managed database service with automatic encryption and scale-up capability provides the best solution for the company's requirements. Managed database services such as Amazon RDS, Google Cloud SQL, and Azure Database offer automatic encryption of data at rest and in transit, and they provide built-in scalability features that can automatically scale up or down the database depending on the workload. This solution also reduces the operational burden of managing a database and ensures that the database is always up to date with the latest security patches."
  },
  {
    Question: "Refer to the EHR Healthcare case study for this question: https://services.google.com/fh/files/blogs/master_case_study_ehr_healthcare.pdf  The sales employees of EHR work remotely and travel to various locations for their job. These employees require access to web-based sales tools located in the EHR data center. EHR has made the decision to retire their existing Virtual Private Network (VPN) infrastructure, necessitating the migration of the web-based sales tools to a BeyondCorp access model. Each sales employee possesses a Google Workspace account, which they utilize for single sign-on (SSO) purposes. What should you do?",
    OptionA: "You should create a Google group for the sales tool application, and upgrade that group to a security group.",
    OptionB: "For every sales employee who needs access to the sales tool application, you should give their Google Workspace user account the predefined AppEngine Viewer role.",
    OptionC: "You should deploy an external HTTP(S) load balancer and create a custom Cloud Armor policy for the sales tool application.",
    OptionD: "You should create an Identity-Aware Proxy (IAP) connector that points to the sales tool application.",
    CorrectAnswer: "You should create an Identity-Aware Proxy (IAP) connector that points to the sales tool application.",
    Discription: "You should create an Identity-Aware Proxy (IAP) connector that points to the sales tool application. -> Correct."
  },
  {
    Question: "A Game Development company deployed a horror game using App Engine in the europe-central2 region. After a while, they see that most of their users live in Japan. They want to minimize latency. As a cloud architect, what should you advise them?",
    OptionA: "They should update the default region to asia-northeast1 in the App Engine.",
    OptionB: "They should deploy a new app engine application in the same GCP project and set the region to asia-northeast1. Finally, remove the old App Engine application.",
    OptionC: "They should move this application deployment to asia-northeast1 region and create a new GCP project. Than, they should create a new App Engine application in the new GCP project and set its region to asia-northeast1. Finally, they should remove the old App Engine application.",
    OptionD: "They should create a ticket to Google Support to change application deployment region in App Engine.",
    CorrectAnswer: "They should move this application deployment to asia-northeast1 region and create a new GCP project. Than, they should create a new App Engine application in the new GCP project and set its region to asia-northeast1. Finally, they should remove the old App Engine application.",
    Discription: "They should move this application deployment to asia-northeast1 region and create a new GCP project. Than, they should create a new App Engine application in the new GCP project and set its region to asia-northeast1. Finally, they should remove the old App Engine application. -> Correct. Latency is the delay between a user action and the application's response. It can be affected by the distance between the user and the application's servers. In this scenario, the majority of the game's users are located in Japan, and the game is currently deployed in the europe-central2 region. Therefore, moving the deployment to a closer region such as asia-northeast1 would reduce the latency experienced by the users. This option is the CorrectAnswer because it involves creating a new App Engine application in a new GCP project in the asia-northeast1 region, which will ensure that the application is deployed closer to the users, thereby reducing latency. Once the new application is deployed, the old application can be removed to prevent any confusion or conflicts."
  },
  {
    Question: "A multinational retail company has a large number of physical stores across multiple countries. The company wants to implement a centralized system to monitor the energy consumption of each store in real-time and generate reports to help optimize energy usage. The company wants to use the Google Cloud Platform for this implementation. Which of the following options would be the best solution for this requirement?",
    OptionA: "Use Cloud Pub/Sub to connect the energy meters in each store to the cloud, Cloud Datastore to store energy consumption data, and Cloud Functions to process the data and generate reports.",
    OptionB: "Use Cloud IoT Edge to connect the energy meters in each store to the cloud, Cloud Firestore to store energy consumption data, and Cloud Tasks to process the data and generate reports.",
    OptionC: "Use Cloud IoT Core to connect the energy meters in each store to the cloud, BigQuery to store energy consumption data, and Cloud Dataflow to process the data and generate reports.",
    OptionD: "Use Cloud IoT Edge to connect the energy meters in each store to the cloud, Cloud SQL to store energy consumption data, and Cloud Pub/Sub to process the data and generate reports.",
    CorrectAnswer: "Use Cloud IoT Core to connect the energy meters in each store to the cloud, BigQuery to store energy consumption data, and Cloud Dataflow to process the data and generate reports.",
    Discription: "Use Cloud IoT Core to connect the energy meters in each store to the cloud, BigQuery to store energy consumption data, and Cloud Dataflow to process the data and generate reports. -> Correct. Cloud IoT Core is a fully managed service for securely connecting and managing IoT devices, making it a suitable choice for connecting the energy meters in each store. BigQuery is a highly scalable and cost-effective data warehouse that can handle large volumes of data, making it a good option for storing energy consumption data. Cloud Dataflow is a fully managed data processing service that can handle batch and stream processing, making it suitable for processing real-time energy consumption data and generating reports."
  },
  {
    Question: "Your organization operates a large-scale web application on Google Cloud, and you need to design a solution to analyze logs from the application in near real-time to detect any potential issues or anomalies. You also want to do some transformation of log data before storing. Which of the following approaches should you recommend?",
    OptionA: "Use Cloud Pub/Sub to ingest logs, Cloud Dataflow to transform, and then BigQuery to analyze.",
    OptionB: "Stream logs directly from the application to BigQuery.",
    OptionC: "Use Cloud Functions to process each log entry and send it to BigQuery.",
    OptionD: "Store logs in Cloud Storage, then use Cloud Dataflow to move them to BigQuery for analysis.",
    CorrectAnswer: "Use Cloud Pub/Sub to ingest logs, Cloud Dataflow to transform, and then BigQuery to analyze.",
    Discription: "Use Cloud Pub/Sub to ingest logs, Cloud Dataflow to transform, and then BigQuery to analyze. -> Correct. Using Cloud Pub/Sub for log ingestion, Cloud Dataflow for transformation, and then storing and analyzing in BigQuery is a common pattern for handling and analyzing large-scale, real-time data in Google Cloud."
  },
  {
    Question: "Your company is deploying a new application on GCP that requires global availability and has heavy content, such as video streaming. The application is designed to handle varying levels of traffic, and you expect users from around the world. What type of load balancer would be the best choice?",
    OptionA: "HTTP(S) Load Balancer",
    OptionB: "SSL Proxy Load Balancer",
    OptionC: "Internal HTTP(S) Load Balancer",
    OptionD: "Network Load Balancer",
    CorrectAnswer: "HTTP(S) Load Balancer",
    Discription: "HTTP(S) Load Balancer -> Correct. HTTP(S) Load Balancer is a global, proxy-based load balancer that supports content-based routing. It integrates with Cloud CDN and Google's global edge network, making it suitable for applications with global reach and heavy content like video streaming."
  },
  {
    Question: "As a cloud architect, you are tasked with designing a solution for a social media application that stores user-uploaded images in a Cloud Storage bucket. For each uploaded image, a thumbnail version needs to be generated for displaying in the application. The solution should be highly scalable to handle sudden spikes in uploads, cost-effective, and able to process images as soon as they are uploaded. What architecture would you suggest for this?",
    OptionA: "Use Compute Engine with a startup script to generate thumbnails.",
    OptionB: "Use Cloud Functions triggered by Cloud Storage events to generate thumbnails.",
    OptionC: "Use Kubernetes Engine to constantly check for new images and generate thumbnails.",
    OptionD: "Use App Engine to constantly check for new images and generate thumbnails.",
    CorrectAnswer: "Use Cloud Functions triggered by Cloud Storage events to generate thumbnails.",
    Discription: "Use Cloud Functions triggered by Cloud Storage events to generate thumbnails. -> Correct. Cloud Functions is a serverless execution environment that can automatically run your function in response to events from Cloud Storage, such as when a new image is uploaded. This makes it an ideal choice for creating a scalable, event-driven, and cost-effective solution to generate thumbnails."
  },
  {
    Question: "An e-commerce company has petabytes of customer behavior data stored in private data center. Due to storage limitations in private data center, this company decided to migrate this data to GCP. The data must be available for your analysts, who have strong SQL background. How should you store the data to meet these requirements?",
    OptionA: "You should import data into BigQuery.",
    OptionB: "You should import data into Cloud Datastore.",
    OptionC: "You should import flat files into Cloud Storage.",
    OptionD: "You should import data into Cloud SQL.",
    CorrectAnswer: "You should import data into BigQuery.",
    Discription: "You should import data into BigQuery. -> Correct. BigQuery is a fully-managed, cloud-native data warehouse that enables super-fast SQL queries using the processing power of Google's infrastructure. It can handle petabytes of data and is designed for analyzing large datasets with ease. It is also highly scalable and supports real-time analysis of data. Analysts can use their existing SQL skills to query data stored in BigQuery without the need for any special software or hardware, making it a perfect fit for the requirements of the e-commerce company in this scenario."
  },
  {
    Question: "You have an application with static content that is deployed in US region. What can you do to bring your content closer to European users?",
    OptionA: "You should move the server to Europe.",
    OptionB: "You should distribute static content using Cloud VPN.",
    OptionC: "You should distribute static content using Cloud CDN.",
    OptionD: "You should scale up the size of the web server.",
    CorrectAnswer: "You should distribute static content using Cloud CDN.",
    Discription: "You should distribute static content using Cloud CDN. -> Correct. Using Cloud CDN will cache static content in edge locations worldwide, including in Europe, and deliver it from the nearest location to the user, reducing latency and improving performance."
  },
  {
    Question: "As a cloud architect, you are working with a media company to develop a web application for live streaming events. The application needs to handle high incoming traffic during popular events. It's also important for the company to keep costs as low as possible when there are no live events (low traffic). Which environment of App Engine should you choose for this scenario?",
    OptionA: "App Engine Flexible environment, because it can scale to zero instances when there's no traffic.",
    OptionB: "Both environments would suit this application equally well.",
    OptionC: "App Engine Flexible environment, because it supports third-party software.",
    OptionD: "App Engine Standard environment, because it can scale to zero instances when there's no traffic.",
    CorrectAnswer: "App Engine Standard environment, because it can scale to zero instances when there's no traffic.",
    Discription: "App Engine Standard environment, because it can scale to zero instances when there's no traffic. -> Correct. It automatically scales instances up and down, even down to zero when there's no traffic. This feature will help the company to keep costs low when there are no live events."
  },
  {
    Question: "A large healthcare organization needs to build a secure and scalable platform for storing and processing sensitive patient data. The platform must meet the following requirements: ensure secure storage and processing of sensitive patient data enable fast and efficient data retrieval for medical professionals support real-time data analysis for clinical decision-making enable data sharing with authorized partners while ensuring data privacy and security minimize costs while still providing high performance Which solution would you recommend to meet these requirements?",
    OptionA: "Implementing a custom-built solution using Cloud Pub/Sub for real-time data processing, Bigtable for data storage, and Google Kubernetes Engine (GKE) for deployment and scaling.",
    OptionB: "Implementing a hybrid solution using Cloud SQL for data storage, Compute Engine for data processing, and BigQuery for real-time analytics.",
    OptionC: "Implementing a serverless solution using Cloud Functions for data processing, Cloud Firestore for data storage, and Cloud Pub/Sub for real-time data processing.",
    OptionD: "Implementing a managed solution using Cloud Healthcare API for data storage and processing, Cloud Dataflow for real-time analytics, and Cloud Identity Access Management (IAM) for secure data sharing.",
    CorrectAnswer: "Implementing a managed solution using Cloud Healthcare API for data storage and processing, Cloud Dataflow for real-time analytics, and Cloud Identity Access Management (IAM) for secure data sharing.",
    Discription: "Implementing a managed solution using Cloud Healthcare API for data storage and processing, Cloud Dataflow for real-time analytics, and Cloud Identity Access Management (IAM) for secure data sharing. -> Correct. The Cloud Healthcare API is designed specifically for healthcare data storage and processing, providing secure and compliant handling of sensitive patient data. Cloud Dataflow enables real-time analytics and data processing, allowing for efficient clinical decision-making. Cloud IAM ensures secure data sharing with authorized partners while maintaining data privacy and security. This managed solution offers a comprehensive set of services that meet the requirements of the healthcare organization."
  },
  {
    Question: "One of your applications is deployed to the GKE cluster as a Kubernetes workload with DaemonSets and is gaining popularity. As a cloud architect, you want to add more pods to your workload and want to make sure the cluster scales up and down automatically based on the volume. What should you do?",
    OptionA: "You should enable Horizontal Pod Autoscaling for the Kubernetes deployment.",
    OptionB: "You should create another identical Kubernetes workload and split traffic between the two workloads.",
    OptionC: "You should perform a rolling update to modify machine type to a higher one.",
    OptionD: "You should enable autoscaling on Kubernetes Engine.",
    CorrectAnswer: "You should enable autoscaling on Kubernetes Engine.",
    Discription: "You should enable autoscaling on Kubernetes Engine. -> Correct. Enabling autoscaling on Kubernetes Engine allows the cluster to automatically adjust the number of nodes based on the demand. This means that as the number of pods in the workload increases, the cluster will automatically add more nodes to handle the load. Similarly, if the workload decreases, the cluster will scale down and remove unnecessary nodes, helping to save costs. This is a more efficient solution than manually modifying the machine type or creating another workload."
  },
   {
    Question: "As a cloud architect, you are responsible for preparing a migration strategy. You need to deploy a disaster recovery infrastructure with the same design and configuration as your production environment using Google Cloud. Which topology would you use?",
    OptionA: "Gated ingress topology",
    OptionB: "Mirrored topology",
    OptionC: "Gated egress topology",
    OptionD: "Handover topology",
    CorrectAnswer: "Mirrored topology",
    Discription: "Mirrored topology -> Correct. A mirrored topology is a disaster recovery strategy that involves creating a mirror of the production environment in a separate location. In this strategy, the disaster recovery environment is an exact replica of the production environment, including the same design and configuration. If a disaster occurs in the production environment, traffic can be quickly redirected to the disaster recovery environment."
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
