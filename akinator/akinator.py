# akinator_toy.py
import math

def entropy(probs):
    """엔트로피 H = -sum p log2 p"""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h

class ToyAkinator:
    def __init__(self):
        # 1) 후보 객체 정의
        # 여기서는 동물 몇 개만 예시로 사용
        self.objects = [
            "개",
            "고양이",
            "토끼",
            "독수리",
            "상어",
            "금붕어",
        ]

        # 2) 질문 정의 (key: 내부 사용, text: 실제 출력 문장)
        # 예/아니오로 답할 수 있는 질문만 사용
        self.questions = {
            "is_pet":       "이 동물은 보통 집에서 키우는 반려동물이야?",
            "lives_in_water": "이 동물은 주로 물에서 살아?",
            "can_fly":      "이 동물은 날 수 있어?",
            "has_fur":      "이 동물은 털(모피)이 많아?",
            "is_mammal":    "이 동물은 포유류야?",
            "is_small":     "이 동물은 대체로 작고 귀여운 편이야?",
        }

        # 3) 지식 베이스: 각 객체가 각 질문에 어떻게 대답하는지 (True/False)
        # 현실적으로 논란 있을 수 있지만, 토이 예제로 생각하면 돼.
        self.knowledge = {
            "개": {
                "is_pet": True, "lives_in_water": False, "can_fly": False,
                "has_fur": True, "is_mammal": True, "is_small": False,
            },
            "고양이": {
                "is_pet": True, "lives_in_water": False, "can_fly": False,
                "has_fur": True, "is_mammal": True, "is_small": True,
            },
            "토끼": {
                "is_pet": True, "lives_in_water": False, "can_fly": False,
                "has_fur": True, "is_mammal": True, "is_small": True,
            },
            "독수리": {
                "is_pet": False, "lives_in_water": False, "can_fly": True,
                "has_fur": False, "is_mammal": False, "is_small": False,
            },
            "상어": {
                "is_pet": False, "lives_in_water": True, "can_fly": False,
                "has_fur": False, "is_mammal": False, "is_small": False,
            },
            "금붕어": {
                "is_pet": True, "lives_in_water": True, "can_fly": False,
                "has_fur": False, "is_mammal": False, "is_small": True,
            },
        }

        # 4) 초기 확률: 균일 분포
        n = len(self.objects)
        self.probs = {obj: 1.0 / n for obj in self.objects}

        # 이미 물어본 질문은 다시 안 쓰기 위해 기록
        self.asked_questions = set()

    def expected_entropy_after_question(self, q_key):
        """
        어떤 질문 q를 했을 때,
        사용자가 '예' 또는 '아니오'라고 답할 확률과
        그 후의 엔트로피를 계산해서
        기대 엔트로피 E[H]를 구한다.
        """
        # 예/아니오 두 가지 가지치기
        yes_probs = {}
        no_probs = {}

        for obj, p in self.probs.items():
            answer = self.knowledge[obj][q_key]
            if answer:
                yes_probs[obj] = yes_probs.get(obj, 0.0) + p
            else:
                no_probs[obj] = no_probs.get(obj, 0.0) + p

        # 가지별 총확률 P(yes), P(no)
        p_yes = sum(yes_probs.values())
        p_no = sum(no_probs.values())

        # yes에서의 엔트로피
        h_yes = 0.0
        if p_yes > 0:
            # 조건부 분포로 정규화
            norm_yes = [p / p_yes for p in yes_probs.values()]
            h_yes = entropy(norm_yes)

        # no에서의 엔트로피
        h_no = 0.0
        if p_no > 0:
            norm_no = [p / p_no for p in no_probs.values()]
            h_no = entropy(norm_no)

        # 기대 엔트로피 = P(yes)*H(yes) + P(no)*H(no)
        expected_h = p_yes * h_yes + p_no * h_no
        return expected_h

    def choose_best_question(self):
        """
        아직 안 물어본 질문들 중에서 정보이득이 가장 큰 질문 선택
        """
        # 현재 엔트로피
        current_probs = list(self.probs.values())
        h_before = entropy(current_probs)

        best_q = None
        best_ig = -1.0

        for q_key in self.questions:
            if q_key in self.asked_questions:
                continue

            h_after = self.expected_entropy_after_question(q_key)
            ig = h_before - h_after
            # 정보이득이 클수록 좋은 질문
            if ig > best_ig:
                best_ig = ig
                best_q = q_key

        return best_q, best_ig

    def update_beliefs(self, q_key, user_answer_yes):
        """
        베이즈 업데이트:
        P(obj | answer) ∝ P(answer | obj) * P(obj)
        여기서는 knowledge가 deterministic이므로
        정답이 맞는 객체만 남기고 정규화.
        """
        new_probs = {}
        for obj, p in self.probs.items():
            obj_answer_yes = self.knowledge[obj][q_key]
            if obj_answer_yes == user_answer_yes:
                new_probs[obj] = p  # 그대로 유지
            else:
                # 모순되는 객체는 확률 0
                new_probs[obj] = 0.0

        # 정규화
        total = sum(new_probs.values())
        if total == 0:
            # 완전히 모순된 경우: 그냥 균일 분포로 리셋(방어 코드)
            n = len(self.objects)
            self.probs = {obj: 1.0 / n for obj in self.objects}
        else:
            for obj in new_probs:
                new_probs[obj] /= total
            self.probs = new_probs

    def best_guess(self):
        """
        현재 확률이 가장 높은 후보와 그 확률 반환
        """
        best_obj = None
        best_p = -1.0
        for obj, p in self.probs.items():
            if p > best_p:
                best_p = p
                best_obj = obj
        return best_obj, best_p

    def run(self):
        print("생각한 동물을 맞춰볼게! (예/아니오로만 답해줘. 그만하려면 '종료' 입력)")

        max_questions = 20
        for i in range(1, max_questions + 1):
            # 1) 현재 최고 후보 확인
            guess, prob = self.best_guess()

            # 어느 정도 확률 이상이면 한 번 추측해본다 (예: 0.85 이상)
            if prob > 0.85 and i > 1:
                print(f"\n내 추측: 당신이 생각한 동물은 '{guess}' 인가요? (예/아니오)")
                ans = input("> ").strip()
                if ans in ["예", "ㅇ", "응", "네", "y", "Y"]:
                    print("역시 맞췄다!")
                    return
                else:
                    # 틀렸다면 이 후보는 버리고 계속 진행
                    self.probs[guess] = 0.0
                    # 정규화
                    total = sum(self.probs.values())
                    if total == 0:
                        # 다 틀어졌으면 게임 종료
                        print("더 이상 후보가 없어... 내가 틀린 것 같아.")
                        return
                    for obj in self.probs:
                        self.probs[obj] /= total

            # 2) 다음 질문 고르기
            q_key, ig = self.choose_best_question()
            if q_key is None:
                # 더 이상 물어볼 질문이 없음
                break

            self.asked_questions.add(q_key)
            question_text = self.questions[q_key]
            print(f"\nQ{i}. {question_text} (예/아니오)")
            ans = input("> ").strip()
            if ans == "종료":
                print("게임을 종료할게.")
                return

            user_yes = ans in ["예", "ㅇ", "응", "네", "y", "Y"]
            # 3) 답변 반영해 확률 업데이트
            self.update_beliefs(q_key, user_yes)

        # 질문 다 쓰거나 max_questions 도달 시 최종 추측
        guess, prob = self.best_guess()
        print(f"\n내 최종 추측: '{guess}' (확률 ~{prob:.2f})")
        print("맞았으면 성공, 아니면 다음에 지식을 더 늘려보자!")


if __name__ == "__main__":
    game = ToyAkinator()
    game.run()
