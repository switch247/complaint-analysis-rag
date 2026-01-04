# Business Understanding: Intelligent Complaint Analysis for Financial Services

## Business Objective
CrediTrust Financial is a fast-growing digital finance company serving East African markets through a mobile-first platform. Their offerings span across:
- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

With a user base of over 500,000 and operations expanding into three countries, CrediTrust receives thousands of customer complaints per month through its in-app channels, email, and regulatory reporting portals.

The mission is to spearhead the development of an internal AI tool that transforms this raw, unstructured complaint data into a strategic asset. This tool is designed for internal stakeholders like Product Managers who currently spend hours manually reading complaints to guess at emerging issues. They need a tool that allows them to ask direct questions and get synthesized, evidence-backed answers in seconds.

## Key Performance Indicators (KPIs)
The success of the project will be measured against three KPIs:
1. **Efficiency**: Decrease the time it takes for a Product Manager to identify a major complaint trend from days to minutes.
2. **Accessibility**: Empower non-technical teams (like Support and Compliance) to get answers without needing a data analyst.
3. **Proactivity**: Shift the company from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## Motivation
CrediTrust’s internal teams face serious bottlenecks:
- **Customer Support** is overwhelmed by the volume of incoming complaints.
- **Product Managers** struggle to identify the most frequent or critical issues across products.
- **Compliance & Risk teams** are reactive rather than proactive when it comes to repeated violations or fraud signals.
- **Executives** lack visibility into emerging pain points due to scattered and hard-to-read complaint narratives.

## Proposed Solution
As a Data & AI Engineer, the task is to develop an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to understand customer pain points across five major product categories:
- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

The goal is to build a **Retrieval-Augmented Generation (RAG)** agent that:
1. Allows internal users to ask plain-English questions about customer complaints (e.g., “Why are people unhappy with Credit Cards?”).
2. Uses semantic search (via a vector database like FAISS or ChromaDB) to retrieve the most relevant complaint narratives.
3. Feeds the retrieved narratives into a language model (LLM) that generates concise, insightful answers.
4. Supports multi-product querying, making it possible to filter or compare issues across financial services.
