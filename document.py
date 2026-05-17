import pandas as pd
import json
import re
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

username = "root"
password = ""
host = "127.0.0.1"
port = "3307"
database = "demo_chatbot"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

def clean_value(value):
    if pd.isna(value) or value is None or str(value).strip() == "":
        return "Data Not Available"
    return value


def parse_json_field(json_text):
    try:
        return json.loads(json_text) if json_text else {}
    except:
        return {}


def get_section_type(app_flag):
    if app_flag is None:
        return "Unknown"

    flag = str(app_flag).strip().upper()

    if flag == "" or flag in ["NONE", "NULL"]:
        return "Unknown"

    if flag == "PE":
        return "Summary"

    if "ROW" in flag and "HEAD" in flag:
        return "Header"

    if re.match(r"^\d+(\.\d+)?[A-Z]?$", flag):
        return "Line Item"

    if "-" in flag:
        return "Sub Item"

    return f"Other ({flag})"

def normalize_text(text):
    if not text:
        return "DATA NOT AVAILABLE"
        
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)        
    return text.strip()

def process_chunk(df_chunk):

    documents = []

    for _, row in df_chunk.iterrows():

        # =====================================================
        # CLEAN VALUES
        # =====================================================
        project_id = clean_value(
            row.get("project_excel_id")
        )

        building_name = clean_value(
            row.get("building_name")
        )

        head_name = clean_value(
            row.get("head_name")
        )

        sr_no = clean_value(
            row.get("sr_no")
        )

        remarks = clean_value(
            row.get("remarks")
        )

        basic_rate = clean_value(
            row.get("basic_rate")
        )

        app_flag = clean_value(
            row.get("app_flag")
        )

        inactive = clean_value(
            row.get("inactive")
        )

        # =====================================================
        # JSON PARSING
        # =====================================================
        rate_details = parse_json_field(
            row.get("rate_details")
        )

        amount = clean_value(
            rate_details.get("amount_incl_gst")
        )

        built_rate = clean_value(
            rate_details.get("rate_built_up_area")
        )

        saleable_rate = clean_value(
            rate_details.get("rate_saleable_area")
        )

        # =====================================================
        # SECTION TYPE
        # =====================================================
        section_type = get_section_type(
            app_flag
        )

        # =====================================================
        # STATUS
        # =====================================================
        status = (
            "Inactive"
            if inactive == 1
            else "Active"
        )

        # =====================================================
        # SEARCH KEYWORDS
        # =====================================================
        keywords = f"""
        {head_name}
        construction
        costing
        civil engineering
        project estimation
        building work
        contractor
        structural work
        quantity survey
        project cost
        built up area
        saleable area
        engineering rates
        construction amount
        """.strip()

        # =====================================================
        # DOCUMENT CONTENT
        # =====================================================
        content = f"""
==============================
PROJECT INFORMATION
==============================

Project ID:
{project_id}

Building Name:
{building_name}

Construction Category:
{head_name}

Section Type:
{section_type}

Serial Number:
{sr_no}

Project Status:
{status}


==============================
COSTING DETAILS
==============================

Total Amount Including GST:
{amount}

Built-up Area Rate:
{built_rate}

Saleable Area Rate:
{saleable_rate}

Basic Rate:
{basic_rate}


==============================
ENGINEERING DETAILS
==============================

Remarks:
{remarks}


==============================
SEARCH KEYWORDS
==============================

{keywords}


==============================
DOCUMENT TYPE
==============================

Construction Project Costing Record

This document contains detailed construction costing,
building estimation, engineering calculations,
civil work quantities, contractor costing,
project budgeting, built-up area calculations,
saleable area calculations, and project financial details.
""".strip()

        # =====================================================
        # NORMALIZED VALUES
        # =====================================================
        normalized_building = normalize_text(
            building_name
        )

        normalized_head = normalize_text(
            head_name
        )

        normalized_section = normalize_text(
            section_type
        )

        # =====================================================
        # METADATA
        # =====================================================
        metadata = {

            # ---------------- PRIMARY ----------------
            "id": row.get("id"),

            "project_id": project_id,

            # ---------------- BUILDING ----------------
            "building": normalized_building,

            "original_building": building_name,

            # ---------------- CATEGORY ----------------
            "head": normalized_head,

            "original_head": head_name,

            # ---------------- SECTION ----------------
            "section": normalized_section,

            "original_section": section_type,

            # ---------------- NUMBERS ----------------
            "sr_no": sr_no,

            "amount": amount,

            "built_rate": built_rate,

            "saleable_rate": saleable_rate,

            "basic_rate": basic_rate,

            # ---------------- STATUS ----------------
            "inactive": inactive,

            "status": status,

            # ---------------- EXTRA ----------------
            "remarks": remarks,

            "document_type":
            "construction_project_record"
        }

        # =====================================================
        # DOCUMENT
        # =====================================================
        doc = Document(
            page_content=content,
            metadata=metadata
        )

        documents.append(doc)

    return documents

BATCH_SIZE = 1000

query = "SELECT * FROM project_top_sheets"

df_iterator = pd.read_sql(query, engine, chunksize=BATCH_SIZE)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


vectorstore = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

import os

if os.path.exists("faiss_index"):
    print("✅ FAISS index already exists. Skipping rebuild.")
    exit()

for batch_num, df_chunk in enumerate(df_iterator):

    print(f"\n🚀 Processing Batch {batch_num + 1}")

    # Step 1: Convert to Documents
    documents = process_chunk(df_chunk)

    # Step 2: Split Documents
    split_docs = text_splitter.split_documents(documents)

    # Step 3: Store in Vector DB
    if vectorstore is None:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    else:
        vectorstore.add_documents(split_docs)

    print(f"✅ Batch {batch_num + 1} Completed")


vectorstore.save_local("faiss_index")
print("✅ Index saved successfully")

