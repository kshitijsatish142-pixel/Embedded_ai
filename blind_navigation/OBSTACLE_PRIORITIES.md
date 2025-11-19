# Obstacle Detection Priorities for Blind Navigation

## Critical Objects (High Priority)

These objects pose immediate safety risks and should be detected with highest priority:

### 1. **Vehicles**
- **Cars, Buses, Trucks, Motorcycles**
- **Why**: High collision risk, moving objects
- **Detection Range**: 10-50 meters
- **Alert Type**: Immediate warning with direction

### 2. **People**
- **Pedestrians, Cyclists, Runners**
- **Why**: Collision risk, need to navigate around
- **Detection Range**: 5-20 meters
- **Alert Type**: Proximity warning

### 3. **Stairs & Steps**
- **Upward steps, Downward steps, Escalators**
- **Why**: Fall risk, navigation critical
- **Detection Range**: 2-10 meters
- **Alert Type**: Direction and height warning

### 4. **Doors**
- **Open doors, Closed doors, Door frames**
- **Why**: Navigation, entry/exit points
- **Detection Range**: 1-5 meters
- **Alert Type**: Status (open/closed) and position

### 5. **Walls & Barriers**
- **Walls, Fences, Railings, Barriers**
- **Why**: Collision prevention
- **Detection Range**: 1-5 meters
- **Alert Type**: Distance and direction

## Important Objects (Medium Priority)

### 6. **Furniture**
- **Chairs, Tables, Desks, Sofas**
- **Why**: Indoor navigation, collision risk
- **Detection Range**: 1-3 meters
- **Alert Type**: Type and position

### 7. **Poles & Posts**
- **Traffic poles, Sign posts, Lamp posts**
- **Why**: Head-level collision risk
- **Detection Range**: 2-10 meters
- **Alert Type**: Height and position

### 8. **Curbs & Edges**
- **Sidewalk curbs, Platform edges, Drop-offs**
- **Why**: Fall prevention
- **Detection Range**: 1-3 meters
- **Alert Type**: Height difference warning

### 9. **Obstacles on Ground**
- **Potholes, Debris, Rocks, Branches**
- **Why**: Trip hazards
- **Detection Range**: 0.5-2 meters
- **Alert Type**: Size and position

### 10. **Overhead Hazards**
- **Low branches, Signs, Awnings, Ceilings**
- **Why**: Head collision risk
- **Detection Range**: 1-3 meters
- **Alert Type**: Height warning

## Navigation Aids (Lower Priority but Useful)

### 11. **Crosswalks**
- **Zebra crossings, Pedestrian crossings**
- **Why**: Safe crossing points
- **Detection Range**: 5-20 meters
- **Alert Type**: Location and direction

### 12. **Traffic Signs**
- **Stop signs, Yield signs, Directional signs**
- **Why**: Navigation assistance
- **Detection Range**: 5-30 meters
- **Alert Type**: Sign type and message

### 13. **Elevators & Escalators**
- **Elevator doors, Escalator entrances**
- **Why**: Building navigation
- **Detection Range**: 2-10 meters
- **Alert Type**: Type and direction

### 14. **Handrails**
- **Stair railings, Platform railings**
- **Why**: Navigation aid, safety
- **Detection Range**: 0.5-2 meters
- **Alert Type**: Position and direction

## Detection Strategy

### Priority Levels
1. **Level 1 (Critical)**: Vehicles, People, Stairs, Doors, Walls
2. **Level 2 (Important)**: Furniture, Poles, Curbs, Ground obstacles, Overhead hazards
3. **Level 3 (Aids)**: Crosswalks, Signs, Elevators, Handrails

### Alert System
- **Immediate**: Critical objects within 3 meters
- **Warning**: Important objects within 2 meters
- **Info**: Navigation aids detected

### Model Architecture Recommendations
- **Lightweight**: MobileNet, YOLOv5n, or EfficientDet-Lite for Pi 4B
- **Real-time**: 10-30 FPS on Pi 4B
- **Multi-class**: Detect all priority objects simultaneously
- **Distance estimation**: Use object size + camera calibration

## Training Data Requirements

### Minimum Dataset Size
- **Critical objects**: 500+ images per class
- **Important objects**: 300+ images per class
- **Navigation aids**: 200+ images per class

### Data Diversity
- **Lighting**: Day, night, dusk, dawn
- **Weather**: Clear, rain, fog
- **Environments**: Indoor, outdoor, urban, residential
- **Angles**: Front, side, overhead views
- **Distances**: Close (1m), medium (5m), far (20m+)

### Annotation Format
- **Bounding boxes**: For all objects
- **Distance labels**: Estimated distance in meters
- **Priority tags**: Level 1, 2, or 3

## Recommended Model Classes

### Phase 1 (MVP - Start Here)
1. Person
2. Vehicle (car, bus, truck)
3. Stairs (up, down)
4. Door (open, closed)
5. Wall/Barrier

### Phase 2 (Enhanced)
Add: Furniture, Poles, Curbs, Ground obstacles

### Phase 3 (Full System)
Add: All navigation aids and remaining objects

## Performance Targets

- **Accuracy**: >85% for critical objects
- **Latency**: <100ms per frame on Pi 4B
- **Range**: Detect critical objects up to 20 meters
- **False Positives**: <5% for critical objects

