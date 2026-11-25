import asyncio, sys
from playwright.async_api import async_playwright
OUT = sys.argv[1]
CSS = """[data-testid=stToolbar],[data-testid=stDecoration],[data-testid=stStatusWidget]{display:none!important}
header[data-testid=stHeader]{background:transparent!important}"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        await pg.goto("http://localhost:8560/", wait_until="networkidle")
        await pg.wait_for_timeout(15000)
        await pg.get_by_text("Suscripción", exact=False).first.click()
        await pg.wait_for_timeout(12000)
        await pg.add_style_tag(content=CSS)
        await pg.screenshot(path=f"{OUT}/10_sub_a.png")
        await pg.evaluate("() => document.querySelector('[data-testid=stMain]').scrollBy(0, 800)")
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=f"{OUT}/10_sub_b.png")
        print(pg.url)
        await b.close()
asyncio.run(main())
