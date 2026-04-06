# OpenEnv Search and Rescue Environment - Complete System Architecture

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Directory Structure](#directory-structure)
5. [Core Components](#core-components)
6. [Data Models](#data-models)
7. [API Specifications](#api-specifications)
8. [Scoring Pipeline](#scoring-pipeline)
9. [Deployment Architecture](#deployment-architecture)
10. [Implementation Checklist](#implementation-checklist)

---

## 🎯 Project Overview

### Mission
Build a production-grade OpenEnv environment for training AI agents in autonomous search and rescue operations in earthquake scenarios.

### Core Objectives
- **Real-world Task**: Simulate earthquake search and rescue operations
- **OpenEnv Compliance**: Full implementation of step()/reset()/state() API with typed models
- **Progressive Difficulty**: 3 difficulty levels (Easy → Medium → Hard)
- **Intelligent Scoring**: Multi-dimensional reward system (Safety, Victim Handling, Decision Intelligence, Efficiency, Time)
- **Deployment Ready**: Dockerized Hugging Face Spaces deployment

### Key Constraints
- **Power Budget**: 90-120 minute battery life
- **Real Sensors**: LiDAR, RGB-D, Thermal, Gas, Acoustic, IMU
- **Triage Priority**: Save at-risk victims first
- **Environmental Challenges**: Weather, debris, structural instability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Agent (External)                          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Policy Net  │  │ Value Net    │  │ Planner      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ Action                             │ Observation
             │                                    │
┌────────────▼────────────────────────────────────▼───────────────┐
│                   OpenEnv API Layer                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  step(action) → (observation, reward, done, truncated)   │  │
│  │  reset() → observation                                    │  │
│  │  state() → StateSnapshot                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
┌────────────▼────────────────────────────────────▼───────────────┐
│                   Environment Core                                │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ World State     │  │ Physics Engine  │  │ Sensor Suite    │ │
│  │ Manager         │  │                 │  │                 │ │
│  │                 │  │ • Collision     │  │ • LiDAR         │ │
│  │ • Robot State   │  │ • Pathfinding   │  │ • RGB-D         │ │
│  │ • Victim State  │  │ • Stability     │  │ • Thermal       │ │
│  │ • Environment   │  │ • Battery Drain │  │ • Gas Sensors   │ │
│  │ • Time/Battery  │  │                 │  │ • Acoustic      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Map Generator   │  │ Victim Generator│  │ Debris System   │ │
│  │                 │  │                 │  │                 │ │
│  │ • Building      │  │ • Demographics  │  │ • Distribution  │ │
│  │   Layout        │  │ • Health Status │  │ • Passability   │ │
│  │ • Zones         │  │ • Entrapment    │  │ • Hazards       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                   Scoring & Evaluation System                    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Multi-Pillar Scoring Engine                              │  │
│  │                                                            │  │
│  │  • Operational Safety (20%)                               │  │
│  │  • Victim Handling (30%)                                  │  │
│  │  • Decision Intelligence (20%)                            │  │
│  │  • Operational Efficiency (20%)                           │  │
│  │  • Time Performance (10%)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Grading System                                           │  │
│  │                                                            │  │
│  │  • Task Success Criteria                                  │  │
│  │  • Penalty Adjudication                                   │  │
│  │  • Metric Aggregation                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

### Episode Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│ EPISODE START                                                     │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ reset()       │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│ Generate Map  │   │ Spawn Victims│   │ Init Robot   │
│               │   │              │   │              │
│ • Buildings   │   │ • Position   │   │ • Location   │
│ • Debris      │   │ • Health     │   │ • Battery    │
│ • Hazards     │   │ • Priority   │   │ • Sensors    │
└───────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Initial Observation  │
                │                      │
                │ • Sensor Readings    │
                │ • Robot Status       │
                │ • Nearby Environment │
                └──────────┬───────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ EPISODE LOOP                                                      │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Agent        │
                    │ Decision     │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ step(action) │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ Action        │  │ World Update │  │ Sensor       │
│ Execution     │  │              │  │ Simulation   │
│               │  │ • Physics    │  │              │
│ • Move        │  │ • Collisions │  │ • LiDAR Scan │
│ • Scan        │  │ • Battery    │  │ • Thermal    │
│ • Rescue      │  │ • Time       │  │ • Gas Level  │
│ • Flag Hazard │  │              │  │ • Acoustic   │
└───────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ State Update     │
                │                  │
                │ • Robot Position │
                │ • Battery Level  │
                │ • Detected Items │
                │ • Map Coverage   │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Reward           │
                │ Calculation      │
                │                  │
                │ 5 Pillars:       │
                │ • Safety         │
                │ • Victim         │
                │ • Decision       │
                │ • Efficiency     │
                │ • Time           │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Termination      │
                │ Check            │
                │                  │
                │ • Time Limit     │
                │ • Battery Dead   │
                │ • Task Complete  │
                │ • Robot Damaged  │
                └─────────┬────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
         ┌──────────┐        ┌──────────┐
         │ Continue │        │ Episode  │
         │ Loop     │        │ End      │
         └────┬─────┘        └────┬─────┘
              │                   │
              └───────────┐       │
                          │       │
                          ▼       ▼
                    ┌──────────────────┐
                    │ Return           │
                    │ (observation,    │
                    │  reward,         │
                    │  done,           │
                    │  truncated,      │
                    │  info)           │
                    └──────────────────┘
```

---

## 📁 Directory Structure

```
rescue-robot-env/
│
├── openenv.yaml                      # OpenEnv specification file
├── README.md                         # Environment documentation
├── LICENSE                           # MIT/Apache 2.0
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container definition
├── .dockerignore
├── .gitignore
│
├── rescue_env/                       # Main package
│   ├── __init__.py
│   │
│   ├── core/                         # Core environment logic
│   │   ├── __init__.py
│   │   ├── environment.py            # Main RescueEnvironment class
│   │   ├── config.py                 # Configuration management
│   │   └── constants.py              # Global constants
│   │
│   ├── models/                       # Pydantic data models
│   │   ├── __init__.py
│   │   ├── actions.py                # Action space models
│   │   ├── observations.py           # Observation space models
│   │   ├── state.py                  # Environment state models
│   │   ├── robot.py                  # Robot specifications
│   │   ├── victim.py                 # Victim models
│   │   ├── environment_params.py     # Level configurations
│   │   └── sensor_data.py            # Sensor output models
│   │
│   ├── world/                        # World simulation
│   │   ├── __init__.py
│   │   ├── map_generator.py          # Procedural map generation
│   │   ├── victim_generator.py       # Victim placement & properties
│   │   ├── debris_system.py          # Debris distribution
│   │   ├── hazard_manager.py         # Gas zones, structural risks
│   │   └── physics_engine.py         # Collision, movement, stability
│   │
│   ├── robot/                        # Robot systems
│   │   ├── __init__.py
│   │   ├── controller.py             # Action execution
│   │   ├── battery.py                # Power management
│   │   ├── sensors/
│   │   │   ├── __init__.py
│   │   │   ├── lidar.py              # LiDAR simulation
│   │   │   ├── camera.py             # RGB-D camera
│   │   │   ├── thermal.py            # Thermal imaging
│   │   │   ├── gas_sensor.py         # Gas detection
│   │   │   ├── acoustic.py           # Sound-based detection
│   │   │   └── imu.py                # Orientation tracking
│   │   └── pathfinding.py            # Navigation algorithms
│   │
│   ├── scoring/                      # Reward & evaluation
│   │   ├── __init__.py
│   │   ├── reward_calculator.py      # Main reward computation
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── safety.py             # Pillar 1: Safety metrics
│   │   │   ├── victim_handling.py    # Pillar 2: Victim metrics
│   │   │   ├── decision.py           # Pillar 3: Decision metrics
│   │   │   ├── efficiency.py         # Pillar 4: Efficiency metrics
│   │   │   └── time_metrics.py       # Pillar 5: Time metrics
│   │   ├── penalties.py              # Absolute penalty system
│   │   └── grader.py                 # Task success evaluation
│   │
│   ├── tasks/                        # Task definitions
│   │   ├── __init__.py
│   │   ├── base_task.py              # Abstract task interface
│   │   ├── easy_task.py              # Easy level configuration
│   │   ├── medium_task.py            # Medium level configuration
│   │   └── hard_task.py              # Hard level configuration
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logger.py                 # Structured logging
│       ├── visualization.py          # Rendering utilities
│       ├── seeding.py                # Random seed management
│       └── validation.py             # Input validation
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_actions.py
│   ├── test_sensors.py
│   ├── test_scoring.py
│   ├── test_map_generation.py
│   └── fixtures/
│       └── sample_configs.py
│
├── baselines/                        # Baseline agents
│   ├── __init__.py
│   ├── random_agent.py               # Random baseline
│   ├── greedy_agent.py               # Greedy triage baseline
│   ├── astar_agent.py                # A* pathfinding baseline
│   └── run_baseline.py               # Evaluation script
│
├── scripts/                          # Utility scripts
│   ├── generate_tasks.py             # Create task instances
│   ├── validate_env.py               # OpenEnv compliance check
│   ├── visualize_episode.py          # Render episodes
│   └── benchmark.py                  # Performance testing
│
├── configs/                          # Configuration files
│   ├── easy.yaml
│   ├── medium.yaml
│   ├── hard.yaml
│   └── robot_specs.yaml
│
├── docs/                             # Documentation
│   ├── ENVIRONMENT.md                # Environment description
│   ├── ACTION_SPACE.md               # Action definitions
│   ├── OBSERVATION_SPACE.md          # Observation structure
│   ├── SCORING.md                    # Reward system details
│   ├── SETUP.md                      # Installation guide
│   └── API.md                        # OpenEnv API reference
│
└── assets/                           # Static assets
    ├── maps/                         # Pre-generated maps (optional)
    ├── visualizations/               # Sample renders
    └── screenshots/                  # README images
```

---

## 🧩 Core Components

### 1. Environment Core (`rescue_env/core/environment.py`)

**Class: `RescueEnvironment`**

```python
class RescueEnvironment:
    """
    Main OpenEnv environment implementing search and rescue simulation.
    
    Attributes:
        difficulty: str - "easy", "medium", or "hard"
        world_state: WorldState - Current simulation state
        robot: Robot - Agent's robot instance
        victims: List[Victim] - All victims in environment
        map: BuildingMap - Generated map structure
        time_elapsed: float - Minutes since episode start
        battery_remaining: float - Percentage (0-100)
        score_tracker: ScoreTracker - Real-time metrics
    """
    
    def __init__(self, difficulty: str = "easy", config: Optional[Dict] = None):
        """Initialize environment with difficulty level."""
        
    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset environment to initial state.
        
        Returns:
            Observation: Initial sensor readings and robot status
        """
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action model (move, scan, rescue, flag, etc.)
            
        Returns:
            observation: Current sensor state
            reward: Float score (0.0 - 1.0 after penalties)
            done: Boolean (task complete or robot destroyed)
            truncated: Boolean (time/battery limit reached)
            info: Dictionary with metrics and debug data
        """
        
    def state(self) -> StateSnapshot:
        """
        Return complete environment state for serialization.
        
        Returns:
            StateSnapshot: Full state including robot, victims, map, metrics
        """
```

**Key Methods:**
- `_generate_world()`: Creates map, places victims, sets hazards
- `_execute_action()`: Processes agent action, updates world
- `_update_physics()`: Collision detection, stability checks
- `_simulate_sensors()`: Generates realistic sensor readings with noise
- `_calculate_reward()`: Computes multi-pillar reward
- `_check_termination()`: Evaluates done/truncated conditions

---

### 2. World Simulation (`rescue_env/world/`)

#### Map Generator (`map_generator.py`)

**Responsibilities:**
- Procedurally generate building layouts based on difficulty
- Create navigable zones with debris obstacles
- Define structural integrity regions
- Place gas hazard zones

**Algorithm:**
```
1. Generate building footprint (grid-based)
2. Assign structural integrity to each cell
3. Simulate earthquake damage:
   - Collapse weak structures
   - Distribute debris based on magnitude
4. Create navigation graph
5. Identify safe zones and hazard zones
```

#### Victim Generator (`victim_generator.py`)

**Responsibilities:**
- Spawn victims with demographic distribution
- Assign health status (Healthy, Vulnerable, Critical)
- Determine entrapment locations
- Calculate survival priority scores

**Priority Formula:**
```
Priority = w1 * health_urgency 
         + w2 * accessibility 
         + w3 * survival_time_remaining
         - w4 * rescue_difficulty
```

#### Physics Engine (`physics_engine.py`)

**Simulations:**
- **Collision Detection**: AABB (Axis-Aligned Bounding Box) with debris
- **Stability Calculation**: Terrain slope, debris weight, robot balance
- **Battery Drain**: Based on movement speed, sensor load, terrain difficulty
- **Movement Constraints**: Speed limits, pathability checks

---

### 3. Robot Systems (`rescue_env/robot/`)

#### Sensor Suite

**LiDAR (`sensors/lidar.py`)**
```python
def scan(position: Position, max_range: float = 30.0) -> PointCloud:
    """
    360° scan returning 3D point cloud.
    
    Noise Model:
    - Easy: 0% noise
    - Medium: 20% noise (fog interference)
    - Hard: 40-60% noise (rain, dust, debris)
    
    Returns:
        PointCloud: Array of (x, y, z, intensity) points
    """
```

**Thermal Camera (`sensors/thermal.py`)**
```python
def detect_heat_signatures(fov: float = 120.0) -> List[ThermalSignature]:
    """
    Detect body heat within field of view.
    
    Returns:
        List of (position, temperature, confidence)
    """
```

**Gas Sensors (`sensors/gas_sensor.py`)**
```python
def measure_gases() -> GasReading:
    """
    Measure: O2, CO, CH4, CO2, H2S levels
    
    Returns:
        GasReading: Concentration levels with safety thresholds
    """
```

**Acoustic Sensor (`sensors/acoustic.py`)**
```python
def listen_for_sounds(duration: float = 5.0) -> List[SoundEvent]:
    """
    Detect victim cries, structural creaks, etc.
    
    Returns:
        List of (type, direction, confidence)
    """
```

---

### 4. Scoring System (`rescue_env/scoring/`)

#### Reward Calculator (`reward_calculator.py`)

**Main Method:**
```python
def calculate_reward(state: WorldState, action: Action, metrics: Metrics) -> float:
    """
    Compute total reward using weighted pillars.
    
    Formula:
        S_final = 0.20 * S_safety 
                + 0.30 * S_victim 
                + 0.20 * S_decision 
                + 0.20 * S_efficiency 
                + 0.10 * S_time
                
        S_adjusted = max(0.0, S_final - sum(penalties))
    
    Returns:
        Float in range [0.0, 1.0]
    """
```

#### Metrics Breakdown

**Safety Metrics (`metrics/safety.py`)**
- `S_survival`: Collision avoidance (base 1.0, penalties per collision)
- `S_stability`: Terrain handling and recovery

**Victim Handling (`metrics/victim_handling.py`)**
- `S_detection`: True positives vs false positives
- `S_location`: Localization accuracy (error thresholds)
- `S_rescue`: Successful rescue rate
- `S_order`: Correct triage priority

**Decision Intelligence (`metrics/decision.py`)**
- `S_priority`: Correct priority assignments
- `S_env`: Useful environment scanning

**Efficiency Metrics (`metrics/efficiency.py`)**
- `S_energy`: Battery utilization vs work done
- `S_path`: Revisit ratio, idle time, smoothness

**Time Performance (`metrics/time_metrics.py`)**
- `S_time`: Golden hour compliance (90/60/45 min limits)

#### Penalty System (`penalties.py`)

**Absolute Penalties:**
```python
PENALTIES = {
    "crush_injury_without_flag": -0.25,
    "critical_gas_zone_entry": -0.15,
    "false_explosion_trigger": -0.30,
    "preventable_destruction": -0.20
}
```

---

## 📊 Data Models

### Action Space (`models/actions.py`)

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Tuple

class ActionType(str, Enum):
    MOVE = "move"
    SCAN_LIDAR = "scan_lidar"
    SCAN_THERMAL = "scan_thermal"
    SCAN_GAS = "scan_gas"
    LISTEN = "listen"
    RESCUE_VICTIM = "rescue_victim"
    FLAG_HAZARD = "flag_hazard"
    IDLE = "idle"

class MoveAction(BaseModel):
    type: ActionType = ActionType.MOVE
    target_position: Tuple[float, float]  # (x, y) in meters
    speed: float = Field(ge=0.0, le=1.6, default=0.8)  # m/s

class ScanAction(BaseModel):
    type: ActionType
    direction: Optional[float] = None  # Angle in degrees, None = 360°
    duration: float = Field(ge=0.1, le=10.0, default=1.0)  # seconds

class RescueAction(BaseModel):
    type: ActionType = ActionType.RESCUE_VICTIM
    victim_id: str
    handling_method: str = "gentle"  # "gentle", "standard", "emergency"

class FlagHazardAction(BaseModel):
    type: ActionType = ActionType.FLAG_HAZARD
    hazard_type: str  # "gas", "structural", "crush_risk"
    location: Tuple[float, float]

# Union type for all actions
Action = MoveAction | ScanAction | RescueAction | FlagHazardAction
```

### Observation Space (`models/observations.py`)

```python
from pydantic import BaseModel
from typing import List, Optional

class RobotStatus(BaseModel):
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: float  # degrees
    battery_level: float  # percentage
    joint_integrity: List[float]  # per-joint health
    is_stable: bool
    carrying_victim: bool

class SensorReadings(BaseModel):
    lidar_points: Optional[List[Tuple[float, float, float, float]]]  # (x,y,z,intensity)
    thermal_signatures: Optional[List[ThermalSignature]]
    gas_levels: Optional[GasReading]
    acoustic_events: Optional[List[SoundEvent]]
    imu_data: IMUReading

class NearbyObjects(BaseModel):
    victims: List[VictimInfo]  # Detected victims in sensor range
    debris: List[DebrisInfo]
    hazards: List[HazardInfo]

class Observation(BaseModel):
    robot_status: RobotStatus
    sensors: SensorReadings
    nearby: NearbyObjects
    time_remaining: float  # minutes
    mission_progress: float  # 0.0 - 1.0
```

### State Snapshot (`models/state.py`)

```python
class StateSnapshot(BaseModel):
    """
    Complete environment state for serialization/checkpointing.
    """
    episode_id: str
    difficulty: str
    timestamp: float
    
    # World state
    map_data: BuildingMap
    all_victims: List[Victim]
    debris_layout: DebrisLayout
    hazard_zones: List[HazardZone]
    
    # Robot state
    robot: Robot
    
    # Metrics
    current_score: ScoreBreakdown
    metrics_history: List[Metrics]
    
    # Episode tracking
    steps_taken: int
    time_elapsed: float
    battery_used: float
    victims_rescued: int
    penalties_incurred: List[PenaltyEvent]
```

---

## 🔌 API Specifications

### OpenEnv YAML (`openenv.yaml`)

```yaml
name: rescue-robot-earthquake-v1
version: 1.0.0
description: |
  Autonomous search and rescue in earthquake disaster scenarios.
  Train AI agents to locate, triage, and rescue victims while managing
  battery life, sensor resources, and environmental hazards.

author: Your Name
license: MIT
homepage: https://huggingface.co/spaces/yourname/rescue-robot-env

environment:
  class: rescue_env.core.RescueEnvironment
  
observation_space:
  type: dict
  spaces:
    robot_status:
      type: dict
      spaces:
        position: {type: box, shape: [3], low: 0, high: 100}
        battery_level: {type: box, shape: [1], low: 0, high: 100}
        is_stable: {type: discrete, n: 2}
    sensors:
      type: dict
      spaces:
        lidar_points: {type: box, shape: [null, 4], dtype: float32}
        thermal_signatures: {type: box, shape: [null, 3], dtype: float32}
        gas_levels: {type: box, shape: [6], low: 0, high: 100}

action_space:
  type: dict
  spaces:
    action_type: {type: discrete, n: 8}
    parameters: {type: box, shape: [10], low: -100, high: 100}

reward_range: [0.0, 1.0]

tasks:
  - name: sweep_and_map
    difficulty: easy
    description: Complete area sweep in stable conditions
    success_criteria:
      map_coverage: 0.95
      victims_detected: 0.90
      battery_remaining: 0.10
    
  - name: strategic_triage
    difficulty: medium
    description: Rescue prioritized victims in fog conditions
    success_criteria:
      critical_victims_rescued: 0.80
      priority_score: 0.75
      safety_score: 0.70
    
  - name: extreme_rescue
    difficulty: hard
    description: Mass casualty event with sensor degradation
    success_criteria:
      victims_rescued: 0.60
      decision_score: 0.65
      mission_completion: 0.70

dependencies:
  - numpy>=1.24.0
  - pydantic>=2.0.0
  - scipy>=1.10.0
```

### Step API Contract

**Input:**
```python
action: Action  # Typed action model
```

**Output:**
```python
(
    observation: Observation,     # Current sensor state
    reward: float,                # Range [0.0, 1.0]
    done: bool,                   # True if episode terminated
    truncated: bool,              # True if time/battery limit hit
    info: Dict[str, Any]          # Debug and metrics data
)
```

**Info Dictionary Contents:**
```python
info = {
    "score_breakdown": {
        "safety": 0.85,
        "victim_handling": 0.72,
        "decision": 0.68,
        "efficiency": 0.55,
        "time": 0.90,
        "total": 0.714
    },
    "metrics": {
        "collisions": 2,
        "victims_detected": 15,
        "victims_rescued": 8,
        "false_positives": 1,
        "battery_used": 67.5,
        "time_elapsed": 42.3,
        "map_coverage": 0.78
    },
    "penalties": [
        {"type": "collision", "amount": -0.04, "step": 145},
        {"type": "collision", "amount": -0.08, "step": 289}
    ],
    "reason": "time_limit_reached",  # if done/truncated
    "success": False
}
```

---

## 🎯 Scoring Pipeline

### Execution Flow

```
Step Execution
     │
     ├─> Action Processing
     │   ├─> Validate action
     │   ├─> Execute in physics engine
     │   └─> Update world state
     │
     ├─> Metrics Collection
     │   ├─> Track collisions
     │   ├─> Record victim interactions
     │   ├─> Monitor battery usage
     │   ├─> Log decision quality
     │   └─> Update coverage map
     │
     ├─> Reward Calculation
     │   │
     │   ├─> Pillar 1: Safety (20%)
     │   │   ├─> S_survival = f(collisions, damage)
     │   │   └─> S_stability = f(instability_events, recoveries)
     │   │
     │   ├─> Pillar 2: Victim Handling (30%)
     │   │   ├─> S_detection = f(TP, FP, confidence)
     │   │   ├─> S_location = f(accuracy, precision)
     │   │   ├─> S_rescue = f(rescued, attempts)
     │   │   └─> S_order = f(priority_adherence)
     │   │
     │   ├─> Pillar 3: Decision Intelligence (20%)
     │   │   ├─> S_priority = f(correct_assignments)
     │   │   └─> S_env = f(useful_scans, coverage)
     │   │
     │   ├─> Pillar 4: Efficiency (20%)
     │   │   ├─> S_energy = f(battery, work_done)
     │   │   └─> S_path = f(revisits, idle, smoothness)
     │   │
     │   └─> Pillar 5: Time (10%)
     │       └─> S_time = f(time_remaining, golden_hour)
     │
     ├─> Penalty Adjudication
     │   ├─> Check crush injuries without flags
     │   ├─> Check gas zone violations
     │   ├─> Check false alarms
     │   └─> Check preventable damage
     │
     ├─> Final Score Computation
     │   └─> S_final = max(0.0, weighted_sum - penalties)
     │
     └─> Return (obs, reward, done, truncated, info)
```

### Incremental Reward Strategy

**To enable learning, rewards are computed at each step:**

1. **Delta Rewards**: Compare current state to previous state
   - New victims detected → +0.05 per victim
   - Victim successfully rescued → +0.10 to +0.30 (by priority)
   - Hazard flagged → +0.03
   - Collision → -0.04 to -0.25

2. **Progress Rewards**: Encourage exploration
   - New area explored → +0.002 per cell
   - Battery efficiency → +0.001 per effective action

3. **Terminal Bonus**: Large reward at episode end
   - Full pillar evaluation → up to +1.0
   - Applied only when done=True or truncated=True

---

## 🚀 Deployment Architecture

### Hugging Face Spaces Setup

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY rescue_env/ ./rescue_env/
COPY openenv.yaml .
COPY baselines/ ./baselines/
COPY configs/ ./configs/

# Expose Gradio port
EXPOSE 7860

# Run demo interface
CMD ["python", "baselines/run_baseline.py", "--demo"]
```

**Gradio Interface:**
```python
import gradio as gr
from rescue_env.core import RescueEnvironment

def run_episode(difficulty, agent_type, steps):
    env = RescueEnvironment(difficulty=difficulty)
    # ... run baseline agent ...
    return visualization, metrics

demo = gr.Interface(
    fn=run_episode,
    inputs=[
        gr.Dropdown(["easy", "medium", "hard"], label="Difficulty"),
        gr.Dropdown(["random", "greedy", "astar"], label="Agent"),
        gr.Slider(100, 1000, label="Max Steps")
    ],
    outputs=[
        gr.Image(label="Episode Visualization"),
        gr.JSON(label="Performance Metrics")
    ],
    title="Rescue Robot Environment",
    description="Watch AI agents navigate earthquake scenarios"
)

demo.launch()
```

---

## ✅ Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up project structure and dependencies
- [ ] Implement Pydantic models (actions, observations, state)
- [ ] Create configuration system (easy/medium/hard configs)
- [ ] Build basic environment skeleton (reset/step/state methods)
- [ ] Implement random seed management

### Phase 2: World Simulation (Week 2)
- [ ] Build map generator (procedural building layouts)
- [ ] Implement victim generator (demographics, health, placement)
- [ ] Create debris system (distribution, passability)
- [ ] Develop hazard manager (gas zones, structural risks)
- [ ] Build physics engine (collision, movement, stability)

### Phase 3: Robot Systems (Week 2-3)
- [ ] Implement robot controller (action execution)
- [ ] Build battery system (consumption model)
- [ ] Create LiDAR sensor (point cloud generation + noise)
- [ ] Implement thermal camera (heat signature detection)
- [ ] Build gas sensor array (multi-gas detection)
- [ ] Create acoustic sensor (sound event detection)
- [ ] Implement IMU (orientation tracking)
- [ ] Build pathfinding system (A* with debris awareness)

### Phase 4: Scoring System (Week 3)
- [ ] Implement reward calculator (main aggregation)
- [ ] Build safety metrics (S_survival, S_stability)
- [ ] Build victim handling metrics (detection, location, rescue, order)
- [ ] Build decision metrics (priority, environment scanning)
- [ ] Build efficiency metrics (energy, path efficiency)
- [ ] Build time metrics (golden hour compliance)
- [ ] Implement penalty system (absolute deductions)
- [ ] Create grader (task success evaluation)

### Phase 5: Tasks & Validation (Week 4)
- [ ] Define easy task (sweep and map)
- [ ] Define medium task (strategic triage)
- [ ] Define hard task (extreme rescue)
- [ ] Write unit tests (>80% coverage)
- [ ] Validate OpenEnv compliance
- [ ] Test all three difficulty levels

### Phase 6: Baselines (Week 4)
- [ ] Implement random agent
- [ ] Implement greedy triage agent
- [ ] Implement A* pathfinding agent
- [ ] Run baseline evaluations (10 episodes each)
- [ ] Record reproducible scores
- [ ] Document baseline performance

### Phase 7: Documentation (Week 5)
- [ ] Write comprehensive README
- [ ] Document action space (with examples)
- [ ] Document observation space (with schemas)
- [ ] Document scoring system (with formulas)
- [ ] Create setup guide
- [ ] Write API reference
- [ ] Add visualization examples

### Phase 8: Deployment (Week 5)
- [ ] Create Dockerfile
- [ ] Build Gradio demo interface
- [ ] Test local container build
- [ ] Deploy to Hugging Face Spaces
- [ ] Verify deployment functionality
- [ ] Add demo video/screenshots

### Phase 9: Polish (Week 6)
- [ ] Add episode visualization (matplotlib/pygame)
- [ ] Implement logging system
- [ ] Add performance benchmarks
- [ ] Create contribution guidelines
- [ ] Final testing and bug fixes
- [ ] Release v1.0.0

---

## 📈 Expected Baseline Performance

| Difficulty | Random Agent | Greedy Agent | A* Agent | Target (Trained) |
|------------|--------------|--------------|----------|------------------|
| Easy       | 0.15 - 0.25  | 0.45 - 0.60  | 0.55 - 0.70 | >0.80           |
| Medium     | 0.10 - 0.20  | 0.30 - 0.45  | 0.40 - 0.55 | >0.70           |
| Hard       | 0.05 - 0.15  | 0.20 - 0.35  | 0.25 - 0.40 | >0.60           |

---

## 🔧 Key Technical Decisions

### 1. **Sensor Noise Modeling**
- Easy: Deterministic sensors (100% accuracy)
- Medium: Gaussian noise on LiDAR/RGB-D (σ = 0.2m)
- Hard: Occlusion + noise + dropouts (40-60% degradation)

### 2. **Map Representation**
- Grid-based occupancy map (0.5m resolution)
- Continuous coordinate system for robot position
- Visibility graph for pathfinding

### 3. **Victim Priority Calculation**
```python
priority = (
    0.40 * health_urgency +       # Critical = 1.0, Vulnerable = 0.6, Healthy = 0.2
    0.30 * (1 - accessibility) +  # Trapped = 1.0, Pinned = 0.6, Free = 0.2
    0.20 * survival_time +        # Hours remaining normalized
    0.10 * detection_confidence   # Sensor certainty
)
```

### 4. **Battery Drain Formula**
```python
drain_per_step = (
    base_movement * speed_factor +
    sensor_active * sensor_load +
    idle * idle_consumption +
    terrain_difficulty * terrain_multiplier
)
```

### 5. **Termination Conditions**
- `done=True`: Robot destroyed OR all victims rescued
- `truncated=True`: Time limit reached OR battery depleted

---

## 🎓 Learning Opportunities

This environment teaches agents to:

1. **Triage Under Uncertainty**: Prioritize victims with incomplete information
2. **Resource Management**: Balance exploration vs exploitation with battery constraints
3. **Sensor Fusion**: Combine LiDAR, thermal, acoustic data for victim detection
4. **Risk Assessment**: Navigate hazardous areas vs. safe routes
5. **Multi-Objective Optimization**: Trade off safety, speed, and completeness
6. **Adaptive Planning**: Adjust strategy as sensors degrade (medium/hard)
7. **Ethical Decision-Making**: Who to save first when resources are limited

---

## 📚 References & Inspiration

- **OpenEnv Specification**: Standard API for RL environments
- **Urban Search and Rescue (USAR)**: Real-world rescue robotics research
- **RoboCup Rescue**: Robot competition for disaster scenarios
- **CRASAR**: Center for Robot-Assisted Search and Rescue
- **NIST Test Methods**: Standard evaluation for rescue robots

---

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines on:
- Adding new sensor types
- Creating custom difficulty levels
- Improving scoring metrics
- Submitting baseline agents

---

**End of System Architecture Document**

This document provides a complete blueprint for implementing the Rescue Robot OpenEnv environment. Each section can be expanded into detailed implementation code following the structure and specifications outlined here.