import pygame
import random
import numpy as np
import torch
import torch.nn as nn

# 초기화
pygame.init()

# 기본 설정
CELL_SIZE = 10
X = CELL_SIZE * 80
Y = CELL_SIZE * 60
screen = pygame.display.set_mode((X, Y),pygame.FULLSCREEN)
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
    def __init__(self, input_size=16, hidden_size=64, output_size=len(COLORS.keys())-1):
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
    def __init__(self, x, y, energy=600, brain=None,lw=50,rs=50,st=50,ls=20):
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
        self.leaf_width = lw # 광합성 양 <-> 물 소모량
        self.root_strength = rs # 물 흡수량 <-> 에너지 소모량
        self.stem_thickness = st # 물, 에너지 저장량 <-> 기관 생성에 소모되는 에너지량
        self.lifespan = ls


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
        self.leaf_width += random.randint(-3,3)
        self.root_strength += random.randint(-3, 3)
        self.stem_thickness += random.randint(-3, 3)
        self.lifespan += random.randint(-3, 3)
        if self.leaf_width < 10: self.leaf_width = 10
        if self.root_strength < 10: self.root_strength = 10
        if self.stem_thickness < 10: self.stem_thickness = 10
        if self.lifespan < 10: self.lifespan = 10
        with torch.no_grad():
            for param in self.brain.parameters():
                if not self.reproduce:
                    mutation = torch.rand_like(param) * random.randint(-(3000-self.age*5),3000-self.age*5) * 0.00005
                else:
                    mutation = torch.rand_like(param) * random.randint(-(2200-self.age*4),2200-self.age*4) * 0.00005
                param.add_(mutation)

    def new_organ(self,x,y,organ,cost=None):
        if cost is None:
            cost=self.stem_thickness
        self.age-=3
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
        # self.age += 1
        if self.age >= self.lifespan*10:
            if random.random() < 0.9 and len(self.parts)>1:
                x, y = random.choice(list(self.parts.keys()))
                new_organism(x,y,600,self.brain,self.leaf_width,self.root_strength,self.stem_thickness,self.lifespan)
            self.death()
            return False

        # 생존 조건
        if self.water <= 0 or self.energy <= 0:
            if random.random() < 0.6 and len(self.parts)>1:
                x, y = random.choice(list(self.parts.keys()))
                new_organism(x,y,600,self.brain,self.leaf_width,self.root_strength,self.stem_thickness,self.lifespan)
            self.death()
            return False

        return True

    def photosynthesis(self,x,y):
        if grid[x][y].organ == LEAF:
            self.energy+=grid[x][y].light * self.leaf_width * 0.1
        else:
            self.energy+=grid[x][y].light * 0.5

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
        elif decision == FLOWER:
            leaves = []
            for (x, y), part in self.parts.items():
                if part == LEAF:
                    leaves.append((x, y))
            (nx, ny) = random.choice(leaves) if leaves else (None, None)
            if nx is not None:
                self.new_organ(nx, ny, FLOWER)
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
            self.water += self.root_depth * self.root_strength
            grid[self.root_pos[0]][self.root_pos[1]].water -= self.root_depth * self.root_strength * 0.1
            self.energy += self.root_depth * 0.1
            grid[self.root_pos[0]][self.root_pos[1]].nutrient -= self.root_depth * 0.1
            # 기관 기능 처리
            for (x, y), organ in list(self.parts.items()):
                if organ == STEM:
                    if sun:
                        self.photosynthesis(x,y)
                elif organ == LEAF:
                    if sun:
                        self.photosynthesis(x,y)
                elif organ == FLOWER:
                    if random.random() < 0.2:
                        self.new_organ(x, y, FRUIT)
                elif organ == VINE:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < X//CELL_SIZE and 0 <= ny < X//CELL_SIZE and grid[nx][ny].organ is not None:
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
            if 0 <= nx < X//CELL_SIZE and 0 <= ny < X//CELL_SIZE and grid[nx][ny].organ is None:
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
            if 0 <= nx < X//CELL_SIZE and 0 <= ny < X//CELL_SIZE:
                adjacents.append((nx, ny))
        return adjacents

def new_organism(x, y, energy=600, brain=None,lw=50,rs=50,st=50,ls=20):
    organism=Organism(x, y, energy, brain,lw,rs,st,ls)
    if brain is not None:
        organism.mutate()
    organisms.append(organism)

def get_organism_by_pos(x,y):
    for org in organisms:
        if (x,y) in list(org.parts.keys()):
            return org
    return None

# 격자 초기화
grid = [[Cell() for _ in range(Y)] for _ in range(X)]
organisms = []

# 초기 씨앗 배치
for _ in range(100):
    x = random.randint(0, X//CELL_SIZE - 1)
    y = random.randint(0, Y//CELL_SIZE - 1)
    if grid[x][y].organ is None:
        new_organism(x, y, 600)

time_count = 200
sun=True

# seed_cool = 2000

# 메인 루프
running = True
speed=20
paused=False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_UP:
                # 속도 증가
                speed = min(speed + 10, 201)
                print(speed)
            elif event.key == pygame.K_DOWN:
                # 속도 감소
                speed = max(speed - 10, 1)
                print(speed)
            elif event.key == pygame.K_RETURN:
                for _ in range(100):
                    x = random.randint(0, X//CELL_SIZE - 1)
                    y = random.randint(0, Y//CELL_SIZE - 1)
                    if grid[x][y].organ is None:
                        new_organism(x, y, 600)

    if paused: continue

    if len(organisms) < 10:
        for _ in range(100):
            x = random.randint(0, X//CELL_SIZE - 1)
            y = random.randint(0, Y//CELL_SIZE - 1)
            if grid[x][y].organ is None:
                new_organism(x, y, 600)

    # print('개체 수: ',len(organisms))
    # print('평균 에너지: ',sum(list(map(lambda x: x.energy, organisms)))/len(organisms))
    # print('평균 수분: ',sum(list(map(lambda x: x.water, organisms)))/len(organisms))
    # print(sum(list(map(lambda x: x.leaf_width, organisms)))/len(organisms),sum(list(map(lambda x: x.root_strength, organisms)))/len(organisms),sum(list(map(lambda x: x.stem_thickness, organisms)))/len(organisms),sum(list(map(lambda x: x.lifespan, organisms)))/len(organisms))

    # time_count-=1
    if time_count<=0:
        sun=not sun
        time_count=100

    # seed_cool-=1
    # if seed_cool<=0:
    #     for _ in range(100):
    #         x = random.randint(0, X//CELL_SIZE - 1)
    #         y = random.randint(0, Y//CELL_SIZE - 1)
    #         if grid[x][y].organ is None:
    #             new_organism(x, y, 600)
        # seed_cool = 1000

    screen.fill((255, 255, 255) if sun else (100, 100, 100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dead=[]
    for org in organisms:
        org.update(grid)
    for org in dead:
        organisms.remove(org)

    for x in range(X//CELL_SIZE):
        for y in range(Y//CELL_SIZE):
            organ = grid[x][y].organ

            grid[x][y].water = 100000 # test
            grid[x][y].nutrient = 100000 # test

            if organ is not None:
                pygame.draw.rect(screen, COLORS[organ], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(speed)

pygame.quit()
