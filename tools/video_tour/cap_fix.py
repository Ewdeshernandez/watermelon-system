import asyncio, json, os, sys
from playwright.async_api import async_playwright
OUT = sys.argv[1]
CSS = """[data-testid=stToolbar],[data-testid=stDecoration],[data-testid=stStatusWidget]{display:none!important}
header[data-testid=stHeader]{background:transparent!important}"""
SCRUB_TERMS = json.loads(os.environ.get("WM_VIDEO_SCRUB", "[]"))  # textos a quitar (ubicación, etc.)
SCRUB = """(terms) => { if (!terms.length) return; const w=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
 let n; while((n=w.nextNode())){ for (const t of terms) { if (n.nodeValue.includes(t))
   n.nodeValue=n.nodeValue.split(' · '+t).join('').split(t+' · ').join('').split(t).join(''); } } }"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        await pg.goto("http://localhost:8560/", wait_until="networkidle")
        await pg.wait_for_timeout(20000)
        await pg.add_style_tag(content=CSS)
        await pg.evaluate("() => document.querySelector('[data-testid=stMain]').scrollBy(0, 395)")
        await pg.wait_for_timeout(2500); await pg.evaluate(SCRUB, SCRUB_TERMS)
        await pg.screenshot(path=f"{OUT}/02_train.png")
        s = pg.locator("summary", has_text="Overall trend").first
        await s.scroll_into_view_if_needed(); await s.click()
        await pg.wait_for_timeout(8000)
        await pg.get_by_role("button", name="30D", exact=True).first.click()
        await pg.wait_for_timeout(9000)
        await pg.evaluate("""() => { const m=document.querySelector('[data-testid=stMain]');
          const el=[...document.querySelectorAll('summary')].find(e=>e.textContent.includes('Overall trend'));
          m.scrollBy(0, el.getBoundingClientRect().top-20); }""")
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{OUT}/04_trend.png")
        await b.close()
asyncio.run(main())
