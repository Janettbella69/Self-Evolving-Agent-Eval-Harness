# Annotation Guide: Shopping Research Agent

## What You're Evaluating

The agent receives a shopping query and produces a Buyer's Guide with product recommendations, comparisons, prices, and sources. Your job: **would a real user find this guide helpful enough to make a purchase decision?**

## Pass/Fail Definition

### PASS — Guide is usable for purchase decision
- At least 3 relevant products recommended
- Prices are present and roughly accurate (within ~15%)
- Key specs/features are correct (not hallucinated)
- Sources are real and support the claims
- Addresses the user's specific needs/budget/constraints
- Side-by-side comparison exists (even if not perfect)

### FAIL — Guide is NOT usable
- Missing products or wrong category entirely
- Prices are missing, fabricated, or wildly wrong
- Critical features hallucinated (e.g., "waterproof" when it's not)
- Sources are dead links or don't support claims
- Ignores user constraints (budget, use case, preferences)
- Guide is too short (<500 chars) or mostly filler

## Quick Judgment Framework

Ask yourself 3 questions:
1. **Would I trust this to shop with?** If you'd verify everything yourself, it's FAIL.
2. **Are the product claims real?** Spot-check 2-3 specs. If any are fabricated, FAIL.
3. **Does it match what was asked?** If the user asked for "under $100" and top pick is $300, FAIL.

## Examples

### Example 1: PASS
> Query: "best wireless earbuds under $100 for running"
> Guide: Recommends Sony WF-C700N ($78), JLab Epic Air Sport ($60), Jabra Elite 4 ($80). Comparison table with battery, IP rating, ANC. Notes WF-C700N lacks IP rating for sports. Sources: RTINGS, SoundGuys, Amazon.

Why PASS: Products match budget, running use-case addressed (IP rating mentioned), prices accurate, expert sources cited.

### Example 2: FAIL
> Query: "best 4K monitor for photo editing under $500"
> Guide: Recommends Dell U2723QE ($620), ASUS ProArt PA279CRV ($450). Only 2 products. No color accuracy specs. Claims "100% DCI-P3" for Dell without source.

Why FAIL: Dell over budget, only 2 products, critical spec (color gamut) uncited, user constraint violated.

### Example 3: FAIL (borderline)
> Query: "best robot vacuum for pet hair"
> Guide: 5 products with prices and comparison table. But iRobot Roomba j7+ listed at $399 (actual: $599). Sources are all "techradar.com" — no RTINGS or specialized reviews.

Why FAIL: Price 33% wrong (critical for purchase), low source diversity. Close to pass otherwise.

## How to Annotate

1. Open trace in TracePage
2. Read the **Guide** tab (main output)
3. Check **Products** tab (are they real? prices reasonable?)
4. Check **Sources** tab (are URLs real? do domains match claims?)
5. Click **Pass** or **Fail** button in top-right. Machine scores, the Scores/Logs/AI tabs and the grader breakdown stay hidden until you do (blind labeling), so judge from the trace itself
6. (Optional) Score individual graders in the **Human Calibration** panel on the right (≥70 counts as Pass); each grader's LLM score appears after you score it
7. (Optional) Add open codes in **Codes** tab for pattern tracking

## Time Target

Aim for 3-5 minutes per trace. Don't verify every claim — spot-check 2-3.
