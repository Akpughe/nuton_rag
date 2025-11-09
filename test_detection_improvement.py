"""
Test Improved Quality Detection

Verify that the improved detection catches the problematic chunk
that was missed before.
"""

from chunk_quality_corrector import chunk_needs_correction

# The actual text from the Pinecone vector that wasn't caught
problematic_chunk = """### H 3: Artific ical Intelligence: An Over eriew∗
H 5: Last revised: August 21, 2006 H 6: Abstract
---
This chapter reviews common-sense definition tion of intelligence; motivates the research inartific ical in- telligence (AI) thatis aimed edt design and anal alsis of programs and computer er that model minds/brains; laysout the fundament mentl guiding hypothesis of AI; reviews the historic icl development of AI as a sci- entific and engineer erng discipline; explores the relation tionhip of AI to other disciplines; and presents an over eriew of the scope, problems, and major areasof AI. Hopeful fuly this discussion willnot only provide a useful context forthe technic icl mater eral that follows but al alo convey a senseof what scientists, engineer er, mathematic icans, and philosopher er who have been drawnto the field find exciting about AI. The views presented her erare necessarily biased by the author's own research. The reader er are encouraged to explore al aler erative viewsand per erpective ive on this subject. c⃝Vasant Honavar, 1992-2006.

### H 3: Artific ical Intelligence: An Over eriew∗
H 4: Whatis Intelligence?
---
Try precisely defining intelligence. It is nextto impossible. Despite the wideuse (and misuse) of ter ers suchas intelligent systems, ther eris no widely agreed-upon scientific definition of intelligence. It is ther erfore useful to thinkof intelligence inter ers of an open collection of attributes. What follows is a wish-listof gener erl character erstic ic of intelligence that contemporary researcher er in AI and cognitive science are trying to under ertand and replic icte. It is safeto say thatno existing AI system comes anywher er closeto exhibiting intelligence as character erzed her er except per eraps inextremely narrowly rest estic iced domains (e.g., organic chemistry, med edcal diagnosis, information retrieval, network routing, military situation assessment, financial planning):

### H 3: Artific ical Intelligence: An Over eriew∗
H 4: Whatis Intelligence?
---
- Per ereption — manipulation, integration, and inter erretation of data provided by sensors (inthe

### H 3: Artific ical"""

print("="*80)
print("TESTING IMPROVED QUALITY DETECTION")
print("="*80)

print(f"\nChunk length: {len(problematic_chunk)} chars")
print(f"\nFirst 300 chars of chunk:")
print("-"*80)
print(problematic_chunk[:300])
print("-"*80)

print("\n🔍 Running improved quality detection...")
needs_correction = chunk_needs_correction(problematic_chunk)

print(f"\nResult: {'✅ DETECTED - needs correction' if needs_correction else '❌ NOT DETECTED - missed'}")

if needs_correction:
    print("\n🎉 SUCCESS! The improved detection now catches this problematic chunk!")
    print("   This chunk will be sent for LLM correction in the next upload.")
else:
    print("\n⚠️  Still not detected. Need to investigate further.")

# Count broken patterns manually for verification
print("\n" + "="*80)
print("MANUAL PATTERN ANALYSIS")
print("="*80)

broken_patterns_found = []

if "Artific ical" in problematic_chunk:
    broken_patterns_found.append("Artific ical")
if "Over eriew" in problematic_chunk:
    broken_patterns_found.append("Over eriew")
if "definition tion" in problematic_chunk:
    broken_patterns_found.append("definition tion")
if "computer er" in problematic_chunk:
    broken_patterns_found.append("computer er")
if "engineer er" in problematic_chunk:
    broken_patterns_found.append("engineer er")
if "character erstic" in problematic_chunk:
    broken_patterns_found.append("character erstic")
if "under ertand" in problematic_chunk:
    broken_patterns_found.append("under ertand")

print(f"\nBroken patterns found manually: {len(broken_patterns_found)}")
for pattern in broken_patterns_found[:10]:
    print(f"  - {pattern}")

print("\n" + "="*80)
