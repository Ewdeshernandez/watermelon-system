import asyncio, os, sys
from playwright.async_api import async_playwright
OUT = sys.argv[1]
async def clean(pg):
    await pg.add_style_tag(content="nextjs-portal{display:none!important}")
    await pg.evaluate("""() => {
      const w=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT); let n;
      const terms = JSON.parse('__TERMS__');
      while((n=w.nextNode())){ for (const t of terms) if(n.nodeValue.includes(t)) n.nodeValue=n.nodeValue.split(t).join('Turbogenerador'); }
      for (const el of document.querySelectorAll('div,section,p')) {
        if (el.childElementCount<3 && el.textContent.trim().startsWith('Notificaciones bloqueadas')) { el.remove(); break; } }
    }""".replace("__TERMS__", os.environ.get("WM_VIDEO_SCRUB", "[]")))

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
        await pg.goto("http://localhost:3200/", wait_until="networkidle", timeout=180000)
        await pg.wait_for_timeout(4000)
        await clean(pg)
        await pg.screenshot(path=f"{OUT}/11_mob_fleet.png")
        await pg.get_by_text("SGT300B").first.click()
        await pg.wait_for_load_state("networkidle", timeout=180000)
        await pg.wait_for_timeout(6000)
        print(pg.url)
        for i in range(4):
            await clean(pg)
            await pg.screenshot(path=f"{OUT}/11_mob_asset_{i}.png")
            await pg.mouse.wheel(0, 760)
            await pg.wait_for_timeout(2500)
        await b.close()
asyncio.run(main())
