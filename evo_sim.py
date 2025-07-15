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

class Brain(nn.Module):
    def __init__(self, input_size=11, hidden_size=16, output_size=len(COLORS.keys())-1):
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

# 개체 (유기체)
class Organism:
    def __init__(self, x, y, energy=8, brain=None):
        self.parts = {(x, y): SEED}
        self.water = 3
        self.energy = energy  # 기본 에너지
        self.alive = True
        self.seed_timer = 10  # 발아까지 시간
        self.resource_tick = 0  # 에너지/수분 감소 간격
        self.root_depth = 1

        self.brain = Brain() if brain is None else brain
        self.fitness = 0
        self.age = 0

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
        # 돌연변이: 뇌의 가중치를 약간 수정
        with torch.no_grad():
            for param in self.brain.parameters():
                mutation = torch.randn_like(param) * 0.1
                param.add_(mutation)

    def new_organ(self,x,y,organ,cost=2):
        self.parts[(x, y)] = organ
        grid[x][y].organ = organ
        self.energy -= cost

    def clear_cell(self,x,y):
        del self.parts[(x, y)]
        grid[x][y].organ = None

    def update(self, grid):
        # 일정 간격마다 에너지/수분 감소
        self.resource_tick += 1
        if self.resource_tick >= 2:
            self.resource_tick = 0
            part_count = len(self.parts)
            self.energy -= 0.3 * part_count
            self.water -= 0.1 * part_count

        self.age+=1
        if self.age>=1000:
            self.alive=False
            if random.random()<0.9:
                x,y=random.choice(list(self.parts.keys()))
                new = Organism(x, y, energy=7, brain=self.brain)
                organisms.append(new)
            for (x, y) in list(self.parts.keys()):
                self.clear_cell(x, y)
            return

        # 씨앗 상태인 경우
        if all(org == SEED for org in self.parts.values()):
            self.seed_timer -= 1
            self.energy+=0.3
            self.water+=0.3
            if self.seed_timer <= 0:
                for (x, y) in list(self.parts.keys()):
                    self.new_organ(x,y,STEM,0)

                    # 최소한의 생존을 위한 뿌리와 잎
                    nx,ny=self.find_empty_adjacent(grid,x,y)
                    if nx is not None:
                        self.new_organ(nx,ny,ROOT)
                    nx, ny = self.find_empty_adjacent(grid, x, y)
                    if nx is not None:
                        self.new_organ(nx, ny, LEAF)
                self.mutate()

            return
        else:
            self.water+=self.root_depth
            for (x, y), organ in list(self.parts.items()):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny].organ is VINE:
                        self.energy -= 0.4
                        self.water -= 0.4
                if organ == STEM:
                    if sun: self.energy+=0.2
                elif organ == LEAF:
                    if sun:
                        self.energy += 1
                elif organ == FLOWER:
                    if self.energy > 6:
                        self.new_organ(x, y, FRUIT,5)
                elif organ == VINE:
                    self.energy += 0.2
                    self.water += 0.2
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny].organ is not None:
                            self.energy += 0.4
                            self.water += 0.4
                elif organ == FRUIT:
                    if random.random() < 0.2:
                        # 기존 개체에서 씨앗이 만들어질 때, 새로운 개체로 분리
                        nx, ny = self.find_empty_adjacent(grid, x, y,10)
                        if nx is not None:
                            new_organism = Organism(nx, ny, energy=7,brain=self.brain)
                            organisms.append(new_organism)
                        self.clear_cell(x, y)
                        return

            # 생존 조건
            if self.water <= 0 or self.energy <= 0:
                self.alive = False
                if random.random() < 0.9:
                    x, y = random.choice(list(self.parts.keys()))
                    new = Organism(x, y, energy=7, brain=self.brain)
                    organisms.append(new)
                for (x, y) in list(self.parts.keys()):
                    self.clear_cell(x, y)
                return

            #조직 추가
            parts_val=list(self.parts.values())
            parts_keys = list(self.parts.keys())
            stems=[]
            vines=[]
            decision = self.decide_action()
            vine=False
            if decision is ROOT:
                self.root_depth+=1
                self.energy-=1
            elif decision is VINE:
                for i in range(len(parts_val)):
                    if parts_val[i] == VINE:
                        vines.append(parts_keys[i])
                if vines:
                    xy=[]
                    for x,y in vines:
                        for nx,ny in self.get_adjacents(grid,x,y):
                            if grid[nx][ny].organ is None:
                                xy.append((nx,ny))

                    xy2=[]
                    if xy:
                        for x,y in xy:
                            cnt=0
                            for nx,ny in self.get_adjacents(grid,x,y):
                                if grid[nx][ny].organ is VINE:
                                    cnt+=1
                            if cnt==1:
                                xy2.append((x,y))

                    if xy2:
                        nx,ny=random.choice(xy2)
                    elif xy:
                        nx,ny=random.choice(xy)
                    else:
                        nx,ny=None,None

                    if nx is not None:
                        self.new_organ(nx, ny, decision)
                        vine=True
                        self.energy-=1
            if not vine:
                for i in range(len(parts_val)):
                    if parts_val[i] == STEM:
                        stems.append(parts_keys[i])
                if stems:
                    xy = []
                    for x, y in stems:
                        for nx, ny in self.get_adjacents(grid, x, y):
                            if grid[nx][ny].organ is None:
                                xy.append((nx, ny))

                    xy2 = []
                    if xy:
                        for x, y in xy:
                            cnt = 0
                            for nx, ny in self.get_adjacents(grid, x, y):
                                if grid[nx][ny].organ is STEM:
                                    cnt += 1
                            if cnt == 1:
                                xy2.append((x, y))

                    if xy2:
                        nx, ny = random.choice(xy2)
                    elif xy:
                        nx, ny = random.choice(xy)
                    else:
                        nx, ny = None, None


                    if nx is not None:
                        if decision is not SEED:
                            self.new_organ(nx, ny, decision)
                            self.energy-=1


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

# 격자 초기화
grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
organisms = []

# 초기 씨앗 배치
for _ in range(30):
    x = random.randint(0, GRID_SIZE - 1)
    y = random.randint(0, GRID_SIZE - 1)
    if grid[x][y].organ is None:
        grid[x][y].organ = SEED
        organisms.append(Organism(x, y))

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
        if not org.alive:
            dead.append(org)
        else:
            org.update(grid)
    for org in dead:
        organisms.remove(org)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            organ = grid[x][y].organ
            if organ is not None:
                pygame.draw.rect(screen, COLORS[organ], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()