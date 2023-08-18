from firebase_functions import (https_fn, options, storage_fn)
from firebase_functions.options import MemoryOption
from firebase_admin import initialize_app
from usp.tree import sitemap_tree_for_homepage
from zep_python import (ZepClient)
from llama_index import download_loader
from llama_index import SimpleWebPageReader
# from llama_index.node_parser.extractors import (
#     MetadataExtractor,
#     SummaryExtractor,
#     QuestionsAnsweredExtractor,
#     TitleExtractor,
#     KeywordExtractor,
#     # MetadataFeatureExtractor,
# )
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext, OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex
from llama_index import set_global_service_context
from urllib.parse import urlparse
import weaviate
import openai
import json
import os
from urllib.parse import urlparse, unquote, quote, urljoin
import mysql.connector
import requests
from bs4 import BeautifulSoup

initialize_app()

# constants and global variables
db_config = {
    "host": os.environ.get("SQL_HOST"),
    "port": os.environ.get("SQL_PORT"),
    "user": os.environ.get("SQL_USER"),
    "password": os.environ.get("SQL_PASS"),
    "database": os.environ.get("SQL_DB"),
}
openai.api_key = os.environ.get("OPENAI_API_KEY")
embed_model = OpenAIEmbedding()
llm = OpenAI(model=os.environ.get("AI_MODEL"), temperature=0)
service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
)
set_global_service_context(service_context)
try:
    zep = ZepClient(os.environ.get("ZEP_API_URL"))
    weaviate_client = weaviate.Client(
        url=os.environ.get("WEAVIATE_URL"),
    )
except Exception as e:
    print(e)


def build_tree(urls):
    items = []

    domain_counts = {}
    domain_urls = {}

    for url in urls:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        domain_urls.setdefault(domain, []).append(url)

    for domain, count in domain_counts.items():
        item = {
            "text": domain,
            "url": f"https://{domain}/",
            "count": count,
            "urls": domain_urls.get(domain, [])
        }
        items.append(item)

    return items


def recursive_crawler(url, depth, max_pages):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    visited_urls = set()
    results = []

    def crawl(url, current_depth):
        if current_depth > depth or len(results) >= max_pages or url in visited_urls:
            return
        try:
            response = requests.get(url)
            visited_urls.add(url)
            if response.status_code == 200:
                results.append(url)
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    new_url = link["href"]
                    full_url = urljoin(base_url, new_url)
                    parsed_new_url = urlparse(full_url)
                    if parsed_new_url.netloc == parsed_url.netloc and parsed_new_url.scheme == parsed_url.scheme:
                        crawl(full_url, current_depth + 1)
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    crawl(url, 0)
    return results


@https_fn.on_request(
    region="europe-west3",
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["post"],
    ),
    memory=MemoryOption.GB_1,
    timeout_sec=540,
)
def parse_url(req: https_fn.Request) -> https_fn.Response:
    payload = json.loads(req.data)
    # check if payload contains url
    if "url" not in payload:
        return https_fn.Response({"status": 'error', "message": "url is not provided!"})
    url = payload['url']
    tree = sitemap_tree_for_homepage(url)
    urls = [page.url for page in tree.all_pages()]
    tree_json = json.dumps(build_tree(urls))
    return https_fn.Response(tree_json)


@https_fn.on_request(
    region="europe-west3",
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["post"],
    ),
    memory=MemoryOption.GB_1,
    timeout_sec=540,
)
def recursive_parse_url(req: https_fn.Request) -> https_fn.Response:
    payload = json.loads(req.data)
    # check if payload contains url
    if "url" not in payload:
        return https_fn.Response({"status": "error", "message": "url is not provided!"})
    url = payload["url"]
    depth = payload.get("depth", 2)
    max_pages = payload.get("max_pages", 20)
    urls = recursive_crawler(url, depth, max_pages)
    tree_json = json.dumps(build_tree(urls))
    return https_fn.Response(tree_json)


def get_clean_filename(url):
    parsed_url = urlparse(url)
    clean_filename = os.path.basename(parsed_url.path)
    return unquote(clean_filename)


def extract(text):
    print("text: ", text)
    text = text.split("/")
    client = text[0]
    hub = text[1]
    file_name = text[2]
    return {"client": client, "hub": hub, "file_name": file_name}


def process_file(url, file_name, type):
    print("Loading documents...")
    if type == "web_page":
        documents = SimpleWebPageReader(html_to_text=True).load_data(
            [url]
        )
    else:
        RemoteReader = download_loader("RemoteReader")
        loader = RemoteReader()
        documents = loader.load_data(url=url)
    print(f"Loaded {len(documents)} docs")

    for index, doc in enumerate(documents):
        metadata = doc.metadata
        if (type == "web_page"):
            file_name = get_clean_filename(url)
        metadata['file_name'] = file_name
        documents[index].metadata = metadata
    print(documents[0])

    node_parser = SimpleNodeParser.from_defaults(
        include_metadata=True,
        include_prev_next_rel=True,
    )
    print("Extracting nodes...")
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    print(f"Extracted {len(nodes)} nodes")
    print(nodes[0])
    docs = []
    if type == "document":
        for node in nodes:
            docs.append({
                "node_id": node.id_,
                "file_name": node.metadata["file_name"],
                "page_label": node.metadata["page_label"],
                "source": node.metadata["Source"],
                "text": node.text,
                "hash": node.hash
            })
    else:
        for node in nodes:
            docs.append({
                "node_id": node.id_,
                "file_name": file_name,
                "source": url,
                "text": node.text,
                "hash": node.hash
            })
    return docs


def update_sql(customer_id, cloud_storage_id, theme_id, resource_status, type):
    theme_id = int(theme_id)
    customer_id = int(customer_id)
    cloud_storage_id = str(cloud_storage_id)
    # Connecting to the MySQL database
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        if type == "web_page":
            # Construct the SQL query
            update_query = (
                "UPDATE KnoledgeBaseResourceDetail "
                "SET ResourceStatus = %s "
                "WHERE CustomerId = %s AND CloudStorageId = %s AND ThemeId = %s"
            )
        else:
            # Construct the SQL query
            update_query = (
                "UPDATE KnoledgeBaseResource "
                "SET ResourceStatus = %s "
                "WHERE CustomerId = %s AND CloudStorageId = %s AND ThemeId = %s"
            )
        # Execute the update query
        cursor.execute(update_query, (resource_status, customer_id, cloud_storage_id, theme_id))
        connection.commit()

        print("Update successful")

    except mysql.connector.Error as err:
        print("Error:", err)

    finally:
        # Close the connection and cursor
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def process_resource(url, file_name, client_name, hub_name, cloud_storage_id, type):
    customer_id = int(client_name)
    theme_id = int(hub_name.split("_")[1])
    cloud_storage_id = str(cloud_storage_id)
    try:
        docs = process_file(url, file_name, type)
    except Exception as e:
        print("Error extracting documents from file: ", e)
        update_sql(customer_id, cloud_storage_id, theme_id, resource_status=2, type=type)
        return
    try:
        with weaviate_client.batch() as batch:
            for data_obj in docs:
                batch.add_data_object(
                    data_obj,
                    class_name=hub_name,
                )
        print("Data objects added to Weaviate")
        update_sql(customer_id, cloud_storage_id, theme_id, resource_status=1, type=type)
        return True
    except Exception as e:
        print("Error adding data objects to Weaviate")
        print(e)
        update_sql(customer_id, cloud_storage_id, theme_id, resource_status=2, type=type)
        return False


@storage_fn.on_object_finalized(
    region="europe-west3",
    memory=MemoryOption.GB_1,
    timeout_sec=540,
)
def process_document(event: storage_fn.CloudEvent[storage_fn.StorageObjectData]):
    """When a document is uploaded in the Storage bucket,
    process its content using the process_file function."""

    print("data: ", event.data)
    # get google storage file signed  url

    name = event.data.name
    url = f"https://firebasestorage.googleapis.com/v0/b/knowhubai.appspot.com/o/{quote(name,safe='')}?alt=media"
    data = extract(name)
    file_name = data['file_name']
    client_name = data['client']
    hub_name = data['hub']
    cloud_storage_id = event.data.generation
    type = "document"
    print("client_name: ", client_name)
    print("hub_name: ", hub_name)
    print("cloud_storage_id: ", cloud_storage_id)
    print("file_name: ", file_name)
    print("url: ", url)
    result = process_resource(url, file_name, client_name, hub_name, cloud_storage_id,  type)
    if result:
        print("success")
        return {"status": "success"}
    else:
        print("error")
        return {"status": "error"}


@https_fn.on_request(
    region="europe-west3",
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["post"],
    ),
    memory=MemoryOption.GB_1,
    timeout_sec=540,
)
def process_webpage(req: https_fn.Request) -> https_fn.Response:
    payload = json.loads(req.data)
    # check if payload contains url
    if 'url' not in payload:
        return https_fn.Response(status=500, response="url is not provided!")
    url = payload['url']
    file_name = url
    client_name = payload['client']
    hub_name = payload['hub']
    cloud_storage_id = payload['cloud_storage_id']
    type = "web_page"
    print("file_name: ", file_name)
    print("client_name: ", client_name)
    print("hub_name: ", hub_name)
    print("cloud_storage_id: ", cloud_storage_id)
    result = process_resource(url, file_name, client_name, hub_name, cloud_storage_id,  type)
    if result:
        print("success")
        return https_fn.Response(status=200, response="success")
    else:
        print("error")
        return https_fn.Response(status=500, response="error")


@https_fn.on_request(
    region="europe-west3",
    cors=options.CorsOptions(
        cors_origins="*",
        cors_methods=["post"],
    ),
    memory=MemoryOption.GB_1,
    timeout_sec=540,
)
def chat(req: https_fn.Request) -> https_fn.Response:
    payload = json.loads(req.data)
    try:
        hub_name = payload['hub_name']
        message = payload['message']
    except Exception as e:
        print(e)
        return https_fn.Response(status=500, response=str(e))
    print("hub_name: ", hub_name)
    print("message: ", message)

    vector_store = WeaviateVectorStore(
        weaviate_client=weaviate_client,
        index_name=hub_name
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        vector_store_query_mode="hybrid",
    )
    print("Generating response...")
    try:
        res = query_engine.query(message)
        answer = json.dumps(res.response, indent=2)
        print("response: ", answer)
        return https_fn.Response(answer)
    except Exception as e:
        print(e)
        return https_fn.Response(status=500, response=str(e))
