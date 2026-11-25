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
        await pg.wait_for_timeout(22000)
        await pg.add_style_tag(content=CSS)

        async def shot(name, wait=1500):
            await pg.wait_for_timeout(wait)
            await pg.evaluate(SCRUB, SCRUB_TERMS)
            await pg.screenshot(path=f"{OUT}/{name}.png")
            print("shot", name)

        async def scroll_to_text(txt, offset=24):
            await pg.evaluate("""([t,o]) => { const m=document.querySelector('[data-testid=stMain]');
              const el=[...document.querySelectorAll('summary,div,span,p')].find(e=>e.childElementCount<4 && e.textContent.trim().startsWith(t));
              if(!el) return false; const r=el.getBoundingClientRect(); m.scrollBy(0, r.top-o); return true; }""", [txt, offset])

        async def open_exp(txt):
            s = pg.locator("summary", has_text=txt).first
            await s.scroll_into_view_if_needed()
            await s.click()

        await shot("01_top")
        await scroll_to_text("TREN DE MÁQUINA", 30)
        await shot("02_train", 3000)

        await open_exp("Tabular List")
        await pg.wait_for_timeout(5000)
        await scroll_to_text("Tabular List", 20)
        await shot("03_tabular")

        await open_exp("Overall trend")
        await pg.wait_for_timeout(9000)
        await scroll_to_text("Overall trend", 20)
        await shot("04_trend")

        await open_exp("Event log")
        await pg.wait_for_timeout(6000)
        await scroll_to_text("Event log", 20)
        await shot("05_events")

        # espejo de rotodinámica (ya abierto)
        await pg.get_by_text("Capture 2026", exact=False).first.wait_for(timeout=120000)
        await scroll_to_text("Waveform · Spectrum · Orbit", 20)
        await shot("06_waveform", 4000)
        await pg.get_by_role("button", name="Spectrum", exact=True).last.click()
        await pg.wait_for_timeout(6000)
        await scroll_to_text("Waveform · Spectrum · Orbit", 20)
        await shot("07_spectrum", 2000)
        await pg.get_by_role("button", name="Orbit", exact=True).last.click()
        await pg.wait_for_timeout(8000)
        await scroll_to_text("Waveform · Spectrum · Orbit", 20)
        await shot("08_orbit_a", 2000)
        await pg.evaluate("() => document.querySelector('[data-testid=stMain]').scrollBy(0, 760)")
        await shot("08_orbit_b", 2000)
        await b.close()
asyncio.run(main())
