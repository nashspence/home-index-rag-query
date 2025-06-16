import pytest
from playwright.sync_api import sync_playwright


def test_streamlit_container():
    url = "http://localhost:8501"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_selector("text=Home Index RAG", timeout=60000)
        assert "Home Index RAG" in page.content()
        browser.close()
