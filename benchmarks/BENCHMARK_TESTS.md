# HandFlow Performance Benchmark Test Suite

## Overview
This benchmark compares HandFlow (screen overlay macropad + hand gestures + paper macropad) against traditional keyboard/mouse input across 4 professional workflows.

**Measurement Method:**
- Time each task from start to completion
- Record error rate (mistakes requiring undo/redo)
- Note cognitive load (did you have to recall a shortcut?)
- Run each task 3 times, take average

---

## 1. PROGRAMMER BENCHMARK (VS Code + Terminal)

### Test Set A: HTML/Web Development (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| A1 | Insert complete HTML form with email, password, submit | Type from memory or search snippets | MacroPad: "Form Template" button |
| A2 | Insert responsive image with srcset, alt, loading="lazy" | Type ~80 chars from memory | MacroPad: "Responsive Img" button |
| A3 | Insert input with all attributes (type, name, id, placeholder, required, pattern) | Type from memory, often forget attributes | MacroPad: "Input Full" button |
| A4 | Wrap selection in div with class | Cmd+Shift+P → "Wrap" → type | Gesture: thumb_index_swipe |
| A5 | Insert CSS flexbox centering template | Type 4 properties from memory | MacroPad: "Flex Center" button |

**HTML Templates to Configure:**
```html
<!-- A1: Form Template -->
<form action="" method="POST">
  <input type="email" name="email" id="email" placeholder="Email" required>
  <input type="password" name="password" id="password" placeholder="Password" required minlength="8">
  <button type="submit">Submit</button>
</form>

<!-- A2: Responsive Image -->
<img
  src=""
  srcset=" 480w, 800w, 1200w"
  sizes="(max-width: 600px) 480px, (max-width: 1000px) 800px, 1200px"
  alt=""
  loading="lazy"
  decoding="async"
>

<!-- A3: Full Input -->
<input
  type="text"
  name=""
  id=""
  placeholder=""
  required
  pattern=""
  minlength=""
  maxlength=""
  aria-label=""
>

<!-- A5: Flex Center -->
display: flex;
justify-content: center;
align-items: center;
gap: 1rem;
```

### Test Set B: OpenCV/Python Development (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| B1 | Insert cv2.rectangle with all params | Type from memory/docs | MacroPad: "CV Rect" button |
| B2 | Insert cv2.putText with font, scale, color | Remember 8 parameters | MacroPad: "CV Text" button |
| B3 | Insert VideoCapture + read loop template | Type 10+ lines | MacroPad: "CV Video Loop" button |
| B4 | Insert contour detection template | 5 lines, specific order | MacroPad: "CV Contours" button |
| B5 | Insert ArUco detection template | Complex, rarely memorized | MacroPad: "CV ArUco" button |

**Python Templates to Configure:**
```python
# B1: Rectangle
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# B2: Put Text
cv2.putText(frame, "text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# B3: Video Loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame here
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# B4: Contour Detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# B5: ArUco Detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)
corners, ids, rejected = detector.detectMarkers(frame)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
```

### Test Set C: Debugging Workflow (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| C1 | Toggle breakpoint on current line | F9 or click gutter | Gesture: pointyclick |
| C2 | Start debugging | F5 | MacroPad: "Debug Start" |
| C3 | Step Over | F10 | Gesture: horizontal_swipe (right) |
| C4 | Step Into | F11 | Gesture: swipeup |
| C5 | Step Out | Shift+F11 | Gesture: thumb_middle_swipe |

### Test Set D: Git Workflow (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| D1 | Stage all + Commit with message | Terminal: git add . && git commit -m "" | MacroPad: "Git Commit" → type message |
| D2 | Pull latest | Terminal: git pull | MacroPad: "Git Pull" |
| D3 | Push to remote | Terminal: git push | MacroPad: "Git Push" |
| D4 | Create new branch | Terminal: git checkout -b name | MacroPad: "Git Branch" |
| D5 | View git diff | Terminal: git diff | MacroPad: "Git Diff" |

### Test Set E: Navigation (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| E1 | Go to definition | Cmd+Click or F12 | Gesture: touch on word |
| E2 | Find all references | Shift+F12 | Gesture: touch_hold on word |
| E3 | Quick file open | Cmd+P | MacroPad: "Quick Open" |
| E4 | Toggle terminal | Cmd+` | MacroPad: "Terminal" |
| E5 | Split editor right | Cmd+\ | MacroPad: "Split" |

---

## 2. 3D DESIGNER BENCHMARK (Fusion 360)

### Test Set A: View Navigation (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| A1 | Orbit view | Middle mouse + drag | Gesture: 5_fingers_close + move |
| A2 | Pan view | Shift + Middle mouse | Gesture: horizontal_swipe |
| A3 | Zoom to fit | Double-click middle mouse | MacroPad: "Fit All" |
| A4 | Front view | View cube click | MacroPad: "Front" |
| A5 | Isometric view | View cube corner | MacroPad: "Iso" |

### Test Set B: Sketch Tools (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| B1 | Line tool | L key | MacroPad: "Line" |
| B2 | Rectangle tool | R key | MacroPad: "Rect" |
| B3 | Circle tool | C key | MacroPad: "Circle" |
| B4 | Dimension tool | D key | MacroPad: "Dim" |
| B5 | Trim tool | T key | MacroPad: "Trim" |

### Test Set C: 3D Operations (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| C1 | Extrude | E key or menu | MacroPad: "Extrude" |
| C2 | Revolve | Menu: Create → Revolve | MacroPad: "Revolve" |
| C3 | Fillet | F key | MacroPad: "Fillet" |
| C4 | Chamfer | Menu: Modify → Chamfer | MacroPad: "Chamfer" |
| C5 | Mirror | Menu: Create → Mirror | MacroPad: "Mirror" |

### Test Set D: Selection & Visibility (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| D1 | Hide selected | H key | Gesture: thumb_index_swipe |
| D2 | Show all | Shift+H or menu | MacroPad: "Show All" |
| D3 | Isolate component | Right-click → Isolate | MacroPad: "Isolate" |
| D4 | Select all in sketch | Ctrl+A | Gesture: 5_fingers_close |
| D5 | Deselect all | Escape | Gesture: horizontal_swipe |

### Test Set E: Complex Workflow - Create Bracket (Timed)

**Task:** Create an L-bracket with:
1. Start new sketch on XY plane
2. Draw L-shape (100mm x 50mm x 10mm thick)
3. Finish sketch
4. Extrude 20mm
5. Add 5mm fillet on inner corner
6. Add 4 mounting holes (8mm diameter)
7. Mirror the bracket

**Measure:** Total time from start to complete model

---

## 3. VIDEO EDITOR BENCHMARK (Final Cut Pro)

### Test Set A: Timeline Navigation (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| A1 | Play/Pause | Space | Gesture: pointyclick |
| A2 | Go to start | Home or Fn+← | MacroPad: "Start" |
| A3 | Go to end | End or Fn+→ | MacroPad: "End" |
| A4 | Next edit point | ↓ key | Gesture: swipeup |
| A5 | Previous edit point | ↑ key | Gesture: thumb_index_swipe |

### Test Set B: Editing Operations (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| B1 | Blade/Cut at playhead | Cmd+B | Gesture: middleclick |
| B2 | Ripple delete | Shift+Delete | MacroPad: "Ripple Del" |
| B3 | Insert clip | W key | MacroPad: "Insert" |
| B4 | Append clip | E key | MacroPad: "Append" |
| B5 | Overwrite clip | D key | MacroPad: "Overwrite" |

### Test Set C: Trimming (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| C1 | Trim start +1 frame | , key | Gesture: horizontal_swipe (left) |
| C2 | Trim start -1 frame | . key | Gesture: horizontal_swipe (right) |
| C3 | Extend edit to playhead | Shift+X | MacroPad: "Extend" |
| C4 | Slip clip | Hold Option + drag | MacroPad: "Slip Mode" |
| C5 | Roll edit | R key + drag | MacroPad: "Roll Mode" |

### Test Set D: Effects & Transitions (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| D1 | Add cross dissolve | Cmd+T | MacroPad: "Dissolve" |
| D2 | Add default transition | Ctrl+T | MacroPad: "Trans" |
| D3 | Color correction panel | Cmd+6 | MacroPad: "Color" |
| D4 | Apply LUT | Effects browser → drag | MacroPad: "LUT" |
| D5 | Keyframe opacity | Click diamond + adjust | Gesture: touch on parameter |

### Test Set E: Audio (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| E1 | Detach audio | Ctrl+Shift+S | MacroPad: "Detach" |
| E2 | Audio fade in | Drag handle | MacroPad: "Fade In" |
| E3 | Audio fade out | Drag handle | MacroPad: "Fade Out" |
| E4 | Mute clip | V key | Gesture: thumb_middle_swipe |
| E5 | Audio inspector | Cmd+4 | MacroPad: "Audio" |

### Test Set F: Complex Workflow - Basic Edit Sequence (Timed)

**Task:** Create 30-second sequence:
1. Import 5 clips
2. Add to timeline in order
3. Trim each clip to ~6 seconds
4. Add cross dissolve between each
5. Add title at start
6. Add fade to black at end
7. Adjust audio levels
8. Export H.264

**Measure:** Total time from import to export start

---

## 4. OFFICE WORKER BENCHMARK (Microsoft Office / Google Workspace)

### Test Set A: Document Navigation (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| A1 | Scroll down one page | Page Down or scroll | Gesture: swipeup |
| A2 | Scroll up one page | Page Up or scroll | Gesture: thumb_index_swipe |
| A3 | Go to document start | Cmd+Home | MacroPad: "Doc Start" |
| A4 | Go to document end | Cmd+End | MacroPad: "Doc End" |
| A5 | Zoom in/out | Cmd+/- or pinch | Gesture: 5_fingers_close |

### Test Set B: Text Formatting (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| B1 | Bold | Cmd+B | MacroPad: "Bold" |
| B2 | Italic | Cmd+I | MacroPad: "Italic" |
| B3 | Underline | Cmd+U | MacroPad: "Underline" |
| B4 | Heading 1 | Cmd+Opt+1 | MacroPad: "H1" |
| B5 | Bullet list | Format menu or remember | MacroPad: "Bullets" |

### Test Set C: Spreadsheet Operations (Excel/Sheets) (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| C1 | Insert SUM formula | Type =SUM() | MacroPad: "SUM" |
| C2 | Insert VLOOKUP | Type formula (complex!) | MacroPad: "VLOOKUP" |
| C3 | Format as currency | Menu or Cmd+Shift+4 | MacroPad: "Currency" |
| C4 | Insert row | Right-click → Insert | MacroPad: "Ins Row" |
| C5 | Delete row | Right-click → Delete | MacroPad: "Del Row" |

**Formula Templates:**
```
=SUM(A1:A10)
=VLOOKUP(lookup_value, table_array, col_index, FALSE)
=IF(condition, true_value, false_value)
=COUNTIF(range, criteria)
=AVERAGE(A1:A10)
```

### Test Set D: Email (Gmail/Outlook) (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| D1 | Compose new email | C or click | MacroPad: "New Email" |
| D2 | Reply | R key | MacroPad: "Reply" |
| D3 | Reply all | A key | MacroPad: "Reply All" |
| D4 | Forward | F key | MacroPad: "Forward" |
| D5 | Send | Cmd+Enter | MacroPad: "Send" |

### Test Set E: Window Management (5 tasks)

| # | Task | Manual Method | HandFlow Method |
|---|------|---------------|-----------------|
| E1 | Switch to next app | Cmd+Tab | Gesture: horizontal_swipe |
| E2 | Minimize window | Cmd+M | MacroPad: "Minimize" |
| E3 | Full screen | Ctrl+Cmd+F | MacroPad: "Fullscreen" |
| E4 | Split screen left | Drag to edge | MacroPad: "Split L" |
| E5 | Split screen right | Drag to edge | MacroPad: "Split R" |

### Test Set F: Complex Workflow - Report Generation (Timed)

**Task:** Create monthly report:
1. Open template document
2. Update title with current month
3. Copy data from spreadsheet (5 tables)
4. Paste and format each table
5. Add 3 charts from spreadsheet
6. Update page numbers
7. Save as PDF
8. Attach to new email

**Measure:** Total time from start to email sent

---

## Benchmark Recording Sheet

### Per-Task Recording:
```
Task ID: ___
Method: [ ] Manual  [ ] HandFlow

Trial 1: ___ seconds  Errors: ___
Trial 2: ___ seconds  Errors: ___
Trial 3: ___ seconds  Errors: ___

Average: ___ seconds
Notes: ________________________________
```

### Summary Table:
```
| Category    | Tasks | Manual Avg | HandFlow Avg | Improvement |
|-------------|-------|------------|--------------|-------------|
| Programming |   25  |    ___s    |     ___s     |    ___%     |
| 3D Design   |   25  |    ___s    |     ___s     |    ___%     |
| Video Edit  |   25  |    ___s    |     ___s     |    ___%     |
| Office      |   25  |    ___s    |     ___s     |    ___%     |
| TOTAL       |  100  |    ___s    |     ___s     |    ___%     |
```

---

## Recommended MacroPad Button Assignments

### Programmer Set (ID: 12)
| Button | Label | Action |
|--------|-------|--------|
| 1 | Debug | F5 |
| 2 | Step | F10 |
| 3 | Into | F11 |
| 4 | Break | F9 |
| 5 | Term | Cmd+` |
| 6 | Git+ | git add . && git commit |
| 7 | Pull | git pull |
| 8 | Push | git push |
| 9 | Snip1 | HTML Form template |
| 10 | Snip2 | CV Rectangle template |
| 11 | Snip3 | Video loop template |
| 12 | Quick | Cmd+P |

### 3D Designer Set (ID: 13)
| Button | Label | Action |
|--------|-------|--------|
| 1 | Extrd | E |
| 2 | Fillet | F |
| 3 | Line | L |
| 4 | Rect | R |
| 5 | Circle | C |
| 6 | Dim | D |
| 7 | Front | Numpad 1 |
| 8 | Top | Numpad 7 |
| 9 | Iso | Numpad 0 |
| 10 | Hide | H |
| 11 | ShowA | Shift+H |
| 12 | Mirror | Menu shortcut |

### Video Editor Set (ID: 14)
| Button | Label | Action |
|--------|-------|--------|
| 1 | Blade | Cmd+B |
| 2 | RipDel | Shift+Delete |
| 3 | Insert | W |
| 4 | Append | E |
| 5 | Diss | Cmd+T |
| 6 | Color | Cmd+6 |
| 7 | Audio | Cmd+4 |
| 8 | Export | Cmd+E |
| 9 | Extend | Shift+X |
| 10 | FadeIn | Custom |
| 11 | FadeOut | Custom |
| 12 | Detach | Ctrl+Shift+S |

### Office Worker Set (ID: 15)
| Button | Label | Action |
|--------|-------|--------|
| 1 | Bold | Cmd+B |
| 2 | Italic | Cmd+I |
| 3 | H1 | Cmd+Opt+1 |
| 4 | Bullet | Custom |
| 5 | SUM | =SUM() template |
| 6 | VLOOK | =VLOOKUP() template |
| 7 | NewMail | C (Gmail) |
| 8 | Send | Cmd+Enter |
| 9 | PDF | Cmd+P → PDF |
| 10 | SplitL | Window snap left |
| 11 | SplitR | Window snap right |
| 12 | Full | Ctrl+Cmd+F |

---

## Expected Results

Based on typical usage patterns, HandFlow should show significant improvement in:

1. **Multi-key shortcuts** (30-50% faster) - Single gesture vs Cmd+Shift+Opt combos
2. **Template insertion** (70-90% faster) - One button vs typing 50+ characters
3. **Context switching** (40-60% faster) - No hand movement from keyboard
4. **Rarely-used features** (50-80% faster) - No menu hunting or shortcut recall
5. **Repetitive actions** (20-40% faster) - Muscle memory for gestures

Areas where manual may be comparable:
- Simple single-key shortcuts (already fast)
- Text-heavy input (typing still required)
- Precise mouse operations (gesture less precise)
