import pygame
import random
import numpy as np
import torch
import torch.nn as nn

# 초기화
pygame.init()

# 기본 설정
GRID_SIZE = 100
CELL_SIZE = 6
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

# 기관 종류
ROOT, STEM, LEAF, FLOWER, FRUIT, SEED, VINE = range(7)

# 색상 정의
COLORS = {
    ROOT: (194, 127, 91),      # 갈색
    STEM: (34, 139, 34),      # 짙은 초록
    LEAF: (0, 255, 0),        # 밝은 초록
    FLOWER: (255, 0, 255),    # 분홍
    FRUIT: (255, 165, 0),     # 주황
    SEED: (200, 200, 200),    # 회색
    VINE: (13, 70, 13),
}
# COLORS = {
#     ROOT: (255,0,255),      # 갈색
#     STEM: (0,0,255),      # 짙은 초록
#     LEAF: (0, 255, 0),        # 밝은 초록
#     FLOWER: (0, 255, 255),    # 분홍
#     FRUIT: (255, 165, 0),     # 주황
#     SEED: (200, 200, 200),    # 회색
#     VINE: (255, 0, 0),
# }

class Brain(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, output_size=len(COLORS.keys())-1):
        super(Brain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.network(x)

# 격자 셀
class Cell:
    def __init__(self):
        self.organ = None
        self.water = 1000000
        self.light = 5
        self.nutrient = 1000000

# 개체 (유기체)
class Organism:
    def __init__(self, x, y, energy=300, brain=None):
        # 상태
        self.parts = {(x, y): SEED}
        self.root_pos = (x,y)
        self.water = 150
        self.energy = energy
        self.brain = Brain() if brain is None else brain
        self.age = -10
        self.reproduce = 0
        self.root_depth = 1
        grid[x][y].organ = SEED

        # 특성
        self.leaf_width = 50 # 광합성 양 <-> 물 소모량
        self.root_strength = 50 # 물 흡수량 <-> 에너지 소모량
        self.stem_thickness = 50 # 물, 에너지 저장량 <-> 기관 생성에 소모되는 에너지량
        self.lifespan = 10


    def get_state(self):
        # 환경 상태 정보
        return torch.tensor([
            self.energy,
            self.water,
            len(self.parts),
            sum(1 for org in self.parts.values() if org == STEM),
            sum(1 for org in self.parts.values() if org == LEAF),
            sum(1 for org in self.parts.values() if org == FLOWER),
            sum(1 for org in self.parts.values() if org == VINE),
            sum(1 for org in self.parts.values() if org == FRUIT),
            sun,
            self.age,
            self.root_depth,
            self.reproduce,
            self.leaf_width,
            self.root_strength,
            self.stem_thickness,
            self.lifespan,
            # random.random()  # 약간의 무작위성 추가
        ], dtype=torch.float32)

    def decide_action(self):
        state = self.get_state()
        with torch.no_grad():
            action_probs = self.brain.forward(state)
        actions=list(COLORS.keys())
        actions.remove(FRUIT)
        return actions[torch.multinomial(action_probs, 1).item()]

    def mutate(self):
        with torch.no_grad():
            for param in self.brain.parameters():
                if not self.reproduce:
                    mutation = torch.rand_like(param) * random.randint(-(3000-self.age*5),3000-self.age*5) * 0.0001
                else:
                    mutation = torch.rand_like(param) * random.randint(-(2200-self.age*4),2200-self.age*4) * 0.0001
                param.add_(mutation)

    def new_organ(self,x,y,organ,cost=None):
        if cost is None:
            cost=self.stem_thickness
        self.parts[(x, y)] = organ
        grid[x][y].organ = organ
        self.energy -= cost

    def clear_cell(self,x,y):
        del self.parts[(x, y)]
        grid[x][y].organ = None

    def survive(self):
        self.energy -= self.root_strength * len(self.parts) * 0.1
        self.water -= self.leaf_width * len(self.parts) * 0.1

        # 노화
        self.age += 1
        if self.age >= self.lifespan*10:
            self.mutate()
            x, y = random.choice(list(self.parts.keys()))
            new_organism(x,y,self.energy,self.brain)
            self.death()
            return False

        # 생존 조건
        if self.water <= 0 or self.energy <= 0:
            if random.random() < 0.6 or (self.reproduce and random.random() < 0.25):
                self.mutate()
                x, y = random.choice(list(self.parts.keys()))
                new_organism(x,y,300,self.brain)
            self.death()
            return False

        return True

    def photosynthesis(self,x,y):
        self.energy+=grid[x][y].light * self.leaf_width * 0.1

    def death(self):
        for (x, y) in list(self.parts.keys()):
            self.clear_cell(x, y)
        dead.append(self)
        return

    def grow(self):
        # 조직 추가
        parts_val = list(self.parts.values())
        parts_keys = list(self.parts.keys())
        stems = []
        vines = []
        nx,ny=None,None
        decision = self.decide_action()
        if decision == ROOT:
            self.root_depth += 1
            # self.energy -= self.stem_thickness * 0.1
        elif decision != SEED:
            for i in range(len(parts_val)):
                if parts_val[i] == STEM or parts_val[i] == ROOT:
                    stems.append(parts_keys[i])
                elif parts_val[i] == VINE:
                    vines.append(parts_keys[i])
            if decision == VINE:
                if vines:
                    xy = []
                    for x, y in vines:
                        for nx, ny in self.get_adjacents(grid, x, y):
                            adj = list(map(lambda x: grid[x[0]][x[1]].organ, self.get_adjacents(grid, nx, ny)))
                            if grid[nx][ny].organ is None and adj.count(VINE) == 1:
                                xy.append((nx, ny))

                    if xy:
                        nx, ny = random.choice(xy)

            if stems and nx is None:
                xy = []
                for x, y in stems:
                    for nx, ny in self.get_adjacents(grid, x, y):
                        adj = list(map(lambda x:grid[x[0]][x[1]].organ,self.get_adjacents(grid, nx, ny)))
                        if grid[nx][ny].organ is None and (adj.count(STEM) + adj.count(ROOT)) == 1:
                            xy.append((nx, ny))

                if xy:
                    nx, ny = random.choice(xy)
                else:
                    nx, ny = None,None

            if nx is not None:
                if decision is not SEED:
                    self.new_organ(nx, ny, decision)

    def update(self, grid):
        survived = self.survive()
        if not survived:
            return

        # 씨앗 상태인 경우
        if all(org == SEED for org in self.parts.values()):
            self.age += 1
            if self.age >= 0:
                (x,y)=self.root_pos
                self.new_organ(x,y,ROOT,0)
            return
        else:
            self.water += self.root_depth * self.root_strength * 0.5
            grid[self.root_pos[0]][self.root_pos[1]].water -= self.root_depth * self.root_strength * 0.1
            self.energy += self.root_depth * 0.1
            grid[self.root_pos[0]][self.root_pos[1]].nutrient -= self.root_depth * 0.1
            # 기관 기능 처리
            for (x, y), organ in list(self.parts.items()):
                if organ == STEM:
                    pass
                elif organ == LEAF:
                    if sun:
                        self.photosynthesis(x,y)
                elif organ == FLOWER:
                    if self.energy > 160:
                        self.new_organ(x, y, FRUIT,150)
                elif organ == VINE:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny].organ is not None:
                            self.energy += self.stem_thickness * 0.1
                            get_organism_by_pos(nx,ny).energy -= self.stem_thickness * 0.1
                elif organ == FRUIT:
                    if random.random() < 0.2:
                        nx, ny = self.find_empty_adjacent(grid, x, y,10)
                        if nx is not None:
                            new_organism(nx, ny)
                        self.clear_cell(x, y)
                        self.reproduce+=1
                        return

            self.grow()


    def find_empty_adjacent(self, grid, x, y, dist=1):
        adjacents=[]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny].organ is None:
                adjacents.append((nx, ny))
        if adjacents:
            if dist==1:
                return random.choice(adjacents)
            else:
                x,y=random.choice(adjacents)
                return self.find_empty_adjacent(grid, x,y, dist-1)
        return None, None

    def get_adjacents(self,grid,x,y):
        adjacents = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                adjacents.append((nx, ny))
        return adjacents

def new_organism(x, y, energy=300, brain=None):
    organisms.append(Organism(x, y, energy=energy, brain=brain))

def get_organism_by_pos(x,y):
    for org in organisms:
        if (x,y) in list(org.parts.keys()):
            return org
    return None

# 격자 초기화
grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
organisms = []

# 초기 씨앗 배치
for _ in range(50):
    x = random.randint(0, GRID_SIZE - 1)
    y = random.randint(0, GRID_SIZE - 1)
    if grid[x][y].organ is None:
        new_organism(x, y, 300)

time_count = 200
sun=True

# 메인 루프
running = True
while running:
    print('개체 수: ',len(organisms))
    # time_count-=1
    if time_count<=0:
        sun=not sun
        time_count=100


    screen.fill((255, 255, 255) if sun else (100, 100, 100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dead=[]
    for org in organisms:
        org.update(grid)
    for org in dead:
        organisms.remove(org)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            organ = grid[x][y].organ
            if organ is not None:
                pygame.draw.rect(screen, COLORS[organ], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(20)

pygame.quit()