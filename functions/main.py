from firebase_functions import https_fn
from firebase_admin import initialize_app
import json
from urllib.parse import urlparse
from usp.tree import sitemap_tree_for_homepage
from typing import Any

initialize_app()


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


@https_fn.on_call()
def parse_sitemap(req: https_fn.CallableRequest) -> Any:
    url = req.data['url']
    tree = sitemap_tree_for_homepage(url)
    urls = [page.url for page in tree.all_pages()]
    tree_json = json.dumps(build_tree(urls))
    return tree_json


@https_fn.on_request()
def parse_url(req: https_fn.Request) -> https_fn.Response:
    payload = json.loads(req.data)
    url = payload['url']
    tree = sitemap_tree_for_homepage(url)
    urls = [page.url for page in tree.all_pages()]
    tree_json = json.dumps(build_tree(urls))
    return https_fn.Response(tree_json)
