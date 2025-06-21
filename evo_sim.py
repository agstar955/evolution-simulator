import pygame
import random

# 초기화
pygame.init()

# 기본 설정
GRID_SIZE = 100
CELL_SIZE = 6
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

# 기관 종류
ROOT, TRUNK, STEM, LEAF, FLOWER, FRUIT, SEED = range(7)

# 색상 정의
COLORS = {
    ROOT: (194, 127, 91),      # 갈색
    TRUNK: (5, 82, 10),
    STEM: (34, 139, 34),      # 짙은 초록
    LEAF: (0, 255, 0),        # 밝은 초록
    FLOWER: (255, 0, 255),    # 분홍
    FRUIT: (255, 165, 0),     # 주황
    SEED: (200, 200, 200),    # 회색
}

# 격자 셀
class Cell:
    def __init__(self):
        self.organ = None

# 개체 (유기체)
class Organism:
    def __init__(self, x, y, energy=8):
        self.parts = {(x, y): SEED}
        self.water = 3
        self.energy = energy  # 기본 에너지
        self.alive = True
        self.seed_timer = 10  # 발아까지 시간
        self.resource_tick = 0  # 에너지/수분 감소 간격

    def new_organ(self,x,y,organ,cost=2):
        self.parts[(x, y)] = organ
        grid[x][y].organ = organ
        self.energy -= cost

    def clear_cell(self,x,y):
        self.parts[(x, y)] = None
        grid[x][y].organ = None

    def update(self, grid):
        # 일정 간격마다 에너지/수분 감소
        self.resource_tick += 1
        if self.resource_tick >= 2:
            self.resource_tick = 0
            part_count = len(self.parts)
            self.energy -= 0.3 * part_count
            self.water -= 0.3 * part_count

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

            return

        # 기관 기능 처리
        for (x, y), organ in list(self.parts.items()):
            if organ == ROOT:
                self.energy += 0.3
                self.water += 1
            elif organ == LEAF:
                if sun:
                    self.energy += 1
            elif organ == TRUNK:
                self.energy -= 0.1
                self.water -= 0.1
            elif organ == FLOWER:
                if self.energy >= 9 and random.random() < 0.3:
                    self.new_organ(x, y, FRUIT,7)
            elif organ == FRUIT:
                if random.random() < 0.2:
                    # 기존 개체에서 씨앗이 만들어질 때, 새로운 개체로 분리
                    nx, ny = self.find_empty_adjacent(grid, x, y,10)
                    if nx is not None:
                        new_organism = Organism(nx, ny, energy=7)
                        organisms.append(new_organism)
                    self.clear_cell(x, y)
                    return

        # 생존 조건
        if self.water <= 0 or self.energy <= 0:
            self.alive = False
            for (x, y) in list(self.parts.keys()):
                self.clear_cell(x, y)
            return

        # 성장 조건
        if self.energy > 4 and random.random() < 0.2:
            parts_val=list(self.parts.values())
            parts_keys = list(self.parts.keys())
            stems=[]
            for i in range(len(parts_val)):
                if parts_val[i] == STEM or parts_val[i] == TRUNK:
                    stems.append(parts_keys[i])
            if stems:
                x, y = random.choice(stems)
                if grid[x][y].organ == STEM:
                    nx, ny = self.find_empty_adjacent(grid, x, y)
                else:
                    nx, ny = self.find_empty_adjacent(grid, x, y,2)
                if nx is not None:
                    self.new_organ(nx, ny, random.choice([ROOT,ROOT,STEM,STEM,STEM,TRUNK,TRUNK,LEAF,LEAF,FLOWER]))

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

# 격자 초기화
grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
organisms = []

# 초기 씨앗 배치
for _ in range(20):
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
    time_count-=1
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
