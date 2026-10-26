"""Browser audit; local fixtures only. Requires Playwright and Chromium."""
import json
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / 'evidence'
OUT.mkdir(exist_ok=True)
ROUTES = ['/', '/accuracy', '/models', '/drift', '/cycles', '/data', '/operations', '/experiments']
errors, results = [], []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=shutil.which('chromium'), args=['--no-sandbox'])
    page = browser.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    for width in [320, 390, 768, 1024, 1440]:
        page.set_viewport_size({'width': width, 'height': 960})
        for route in ROUTES:
            response = page.goto('http://127.0.0.1:3102' + route, wait_until='networkidle')
            assert response.status == 200, route
            overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
            assert not overflow, f'Horizontal page overflow: {route} at {width}'
            results.append({'width': width, 'route': route, 'http_status': 200, 'page_overflow': False})
            if (route == '/' and width in [320, 390, 768, 1440]) or (route in ['/accuracy','/models','/experiments'] and width == 390):
                page.screenshot(path=str(OUT / f'after-{route.strip("/") or "home"}-{width}-synthetic.png'), full_page=True)
    page.goto('http://127.0.0.1:3102/', wait_until='networkidle')
    page.locator('.pipeline-step').nth(2).focus()
    page.keyboard.press('Enter')
    assert 'receipts' in page.locator('#pipeline-detail').inner_text()
    frame = page.locator('.chart-frame').first
    frame.scroll_into_view_if_needed()
    box = frame.bounding_box()
    page.mouse.move(box['x'] + box['width'] * .6, box['y'] + box['height'] * .5)
    page.locator('.recharts-tooltip-wrapper').first.wait_for(state='visible')
    page.get_by_label('Periodo').select_option('6')
    assert page.get_by_label('Periodo').input_value() == '6'
    page.get_by_label('Ventana 6', exact=True).uncheck()
    assert not page.get_by_label('Ventana 6', exact=True).is_checked()
    page.goto('http://127.0.0.1:3102/accuracy', wait_until='networkidle')
    page.get_by_text('Comparar estaciones (todas)', exact=True).click()
    station = page.locator('.station-filter input').first
    station.check()
    assert station.is_checked()
    page.get_by_label('Horizonte', exact=True).select_option(index=1)
    page.goto('http://127.0.0.1:3102/experiments', wait_until='networkidle')
    page.get_by_label('Protocolo', exact=True).select_option('1')
    page.locator('.experiment-rows summary').first.click()
    assert 'learning_rate' in page.locator('.experiment-rows details').first.inner_text()
    page.goto('http://127.0.0.1:3102/', wait_until='networkidle')
    page.emulate_media(reduced_motion='reduce')
    assert page.locator('.pipeline-object').first.evaluate('(el) => getComputedStyle(el).transform') == 'none'
    assert not errors, errors
    browser.close()
report = {'data': 'Controlled synthetic fixtures; not production data', 'viewports_and_routes': results,
          'browser_errors': errors, 'filters_and_pipeline_details': 'passed', 'reduced_motion': 'passed'}
(OUT / 'responsive-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'Passed: {len(results)} viewport/route checks, filters, pipeline details, reduced motion; no page errors.')
