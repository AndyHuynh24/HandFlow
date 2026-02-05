# HandFlow Quick Benchmark Guide

## How to Run
1. Do each task 3 times with Manual method, record times
2. Do each task 3 times with HandFlow, record times
3. Calculate averages and improvement percentage

---

## PROGRAMMER (15 min total)

### Speed Test Tasks (do each 3x)

**Template Insertion (Manual = type from memory, HandFlow = 1 button)**
1. Insert HTML form with email + password + submit
2. Insert cv2.rectangle() with all 5 parameters
3. Insert cv2.VideoCapture read loop (10 lines)

**Debugging Sequence (time entire sequence)**
1. Set breakpoint → Start debug → Step over 3x → Step into → Step out → Stop

**Git Sequence (time entire sequence)**
1. Stage all → Commit "test" → Pull → Push

---

## 3D DESIGNER - Fusion 360 (15 min total)

### Speed Test Tasks (do each 3x)

**View Navigation Sequence**
1. Orbit 360° → Pan left → Zoom to fit → Switch to Front → Switch to Iso

**Create Simple Part**
1. New sketch → Rectangle 50x30 → Finish → Extrude 10mm → Fillet 2mm

**Tool Switching Speed (time switching between)**
1. Line → Rectangle → Circle → Dimension → Trim → Extrude (6 switches)

---

## VIDEO EDITOR - Final Cut Pro (15 min total)

### Speed Test Tasks (do each 3x)

**Edit Sequence**
1. Blade cut → Ripple delete → Insert clip → Add dissolve

**Navigation Speed**
1. Go to start → Play → Stop → Next edit → Previous edit → Go to end

**Complex Task (time entire task)**
1. Import clip → Add to timeline → Trim to 5 sec → Add dissolve at start → Add fade at end → Detach audio

---

## OFFICE WORKER (15 min total)

### Speed Test Tasks (do each 3x)

**Document Formatting**
1. Select text → Bold → Italic → Make Heading 1 → Add bullets

**Spreadsheet**
1. Insert SUM formula → Insert VLOOKUP → Format as currency

**Email + Window Management**
1. Compose email → Switch app → Copy data → Switch back → Paste → Send

---

## Recording Template

```
PROGRAMMER TESTS
================
Task                    | Manual (3 trials)      | HandFlow (3 trials)    | Improvement
HTML Form               | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
CV Rectangle            | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Video Loop              | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Debug Sequence          | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Git Sequence            | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%

3D DESIGNER TESTS
=================
Task                    | Manual (3 trials)      | HandFlow (3 trials)    | Improvement
View Navigation Seq     | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Create Simple Part      | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Tool Switching (6)      | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%

VIDEO EDITOR TESTS
==================
Task                    | Manual (3 trials)      | HandFlow (3 trials)    | Improvement
Edit Sequence           | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Navigation Speed        | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Complex Edit Task       | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%

OFFICE WORKER TESTS
===================
Task                    | Manual (3 trials)      | HandFlow (3 trials)    | Improvement
Document Formatting     | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Spreadsheet Formulas    | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%
Email + Window Mgmt     | ___  ___  ___ = ___avg | ___  ___  ___ = ___avg | ___%

TOTAL SUMMARY
=============
Category        | Manual Avg | HandFlow Avg | Time Saved | Improvement %
----------------|------------|--------------|------------|---------------
Programmer      |            |              |            |
3D Designer     |            |              |            |
Video Editor    |            |              |            |
Office Worker   |            |              |            |
----------------|------------|--------------|------------|---------------
OVERALL         |            |              |            |
```

---

## Key Metrics to Highlight

1. **Template Insertion** - Most dramatic improvement (typing 50+ chars vs 1 button)
2. **Multi-step Sequences** - No hand repositioning between steps
3. **Cognitive Load** - No need to remember complex shortcuts
4. **Error Rate** - Fewer typos/wrong shortcuts with dedicated buttons
