import os
import warnings
from platform import system
from random import choice
from typing import Callable, Any

import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

class VocabularyTrainer:
    def __init__(self, dataset_path: str = 'words.txt'):
        print("Loading AI model and dataset... Please wait.")
        self.encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dataset = self._load_data(dataset_path)
        self.clear_terminal = lambda: os.system('cls' if system() == "Windows" else 'clear')

    def _load_data(self, path: str) -> pd.DataFrame:
        """Loads vocabulary and initializes selection weights."""
        df = pd.read_csv(
            path, 
            sep=' - ', 
            header=None, 
            engine='python', 
            names=['term', 'meaning']
        )
        # Initialize probabilities
        df['weight'] = len(df)
        return df

    def _adjust_weights(self, active_df: pd.DataFrame, selected_items: pd.DataFrame) -> None:
        """Reduces the likelihood of recently seen words appearing immediately."""
        active_df.loc[selected_items.index, 'weight'] = 0
        active_df['weight'] += len(selected_items)

    def _prompt_user(self, valid_options: list[str], render_function: Callable, *args: Any) -> int:
        """Validates user input and forces a re-render on invalid entries."""
        while True:
            user_input = input("\n>> Enter your choice: ").strip()
            if user_input in valid_options:
                return int(user_input)
            self.clear_terminal()
            render_function(*args)

    # ==========================================
    #                 UI MENUS
    # ==========================================
    
    def display_main_menu(self) -> None:
        print("=" * 40)
        print("          VOCABULARY TRAINER          ")
        print("=" * 40)
        print(" [1] Explore Random Terms")
        print(" [2] Fill-in-the-Blank Drill")
        print(" [3] Multiple Choice Quiz")
        print(" [0] Exit Application\n")

    def display_random_term_ui(self, item: pd.Series) -> None:
        print("-" * 40)
        print(f" TERM:    {item.term.values[0]}")
        print(f" MEANING: {item.meaning.values[0]}")
        print("-" * 40)
        print(" [1] Pull another random term")
        print(" [0] Return to Main Menu")

    def display_drill_options(self) -> None:
        print("=" * 40)
        print("           DRILL SETTINGS           ")
        print("=" * 40)
        print(" [1] Provide meanings for given terms")
        print(" [2] Provide terms for given meanings")
        print(" [3] Mixed mode (Both)")
        print(" [0] Return to Main Menu")

    def display_drill_question(self, item: pd.Series, mode: int) -> None:
        print("-" * 40)
        if mode == 1:
            print(f" Define the following term: '{item.term.values[0]}'")
        else:
            print(f" Which term matches this meaning? '{item.meaning.values[0]}'")
        print("-" * 40)

    def display_drill_feedback(self, user_answer: str, item: pd.Series, mode: int) -> None:
        print("\n--- RESULTS ---")
        if mode == 1:
            predicted_vec = self.encoder_model.encode([user_answer])
            actual_vec = self.encoder_model.encode(item.meaning.values)
            
            # Calculate cosine similarity safely
            similarity = cosine_similarity(actual_vec, predicted_vec) / (norm(actual_vec - predicted_vec) + 1e-8)
            is_correct = similarity > 0.5
            correct_answer = item.meaning.values[0]
        else:
            is_correct = item.term.values[0].lower() == user_answer.lower()
            correct_answer = item.term.values[0]

        print(" ✅ CORRECT!" if is_correct else " ❌ INCORRECT.")
        print(f" You answered:    {user_answer}")
        print(f" Expected answer: {correct_answer}")
        print("-" * 40)
        print(" [1] Next Question")
        print(" [0] Return to Main Menu")

    def display_quiz_question(self, items: pd.DataFrame, target_index: int) -> None:
        print("-" * 40)
        print(f" Match the meaning: '{items.iloc[target_index].meaning}'")
        print("-" * 40)
        print(f" [1] {items.iloc[0].term}")
        print(f" [2] {items.iloc[1].term}")

    def display_quiz_feedback(self, is_correct: bool) -> None:
        print("\n ✅ CORRECT!" if is_correct else "\n ❌ INCORRECT.")
        print("-" * 40)
        print(" [1] Next Question")
        print(" [0] Return to Main Menu")

    # ==========================================
    #                 GAME MODES
    # ==========================================

    def launch_random_explorer(self, active_df: pd.DataFrame) -> None:
        while True:
            self.clear_terminal()
            selected_item = active_df.sample(1, weights='weight')
            
            self.display_random_term_ui(selected_item)
            choice = self._prompt_user(['0', '1'], self.display_random_term_ui, selected_item)
            
            if choice == 0:
                break
            self._adjust_weights(active_df, selected_item)

    def launch_typing_drill(self, active_df: pd.DataFrame) -> None:
        self.clear_terminal()
        self.display_drill_options()
        mode_choice = self._prompt_user(['0', '1', '2', '3'], self.display_drill_options)
        
        if mode_choice == 0:
            return

        is_mixed_mode = (mode_choice == 3)
        
        while True:
            self.clear_terminal()
            selected_item = active_df.sample(1, weights='weight')
            current_mode = choice([1, 2]) if is_mixed_mode else mode_choice
            
            self.display_drill_question(selected_item, current_mode)
            user_response = input("\n>> Answer: ").strip()
            
            self.display_drill_feedback(user_response, selected_item, current_mode)
            
            # Helper logic to re-render screen cleanly if invalid input is given
            def re_render() -> None:
                self.display_drill_question(selected_item, current_mode)
                print(f"\n>> Answer: {user_response}")
                self.display_drill_feedback(user_response, selected_item, current_mode)

            post_choice = self._prompt_user(['0', '1'], re_render)
            
            if post_choice == 0:
                break
            self._adjust_weights(active_df, selected_item)

    def launch_multiple_choice(self, active_df: pd.DataFrame) -> None:
        while True:
            self.clear_terminal()
            selected_items = active_df.sample(2, weights='weight')
            target_idx = choice([0, 1])
            
            self.display_quiz_question(selected_items, target_idx)
            user_guess = self._prompt_user(['1', '2'], self.display_quiz_question, selected_items, target_idx)
            
            was_correct = (target_idx == user_guess - 1)
            self.display_quiz_feedback(was_correct)
            
            def re_render() -> None:
                self.display_quiz_question(selected_items, target_idx)
                print(f"\n>> Enter your choice: {user_guess}")
                self.display_quiz_feedback(was_correct)

            post_choice = self._prompt_user(['0', '1'], re_render)
            
            if post_choice == 0:
                break
            self._adjust_weights(active_df, selected_items)

    def run(self) -> None:
        """Main execution loop."""
        mode_map = {
            1: self.launch_random_explorer,
            2: self.launch_typing_drill,
            3: self.launch_multiple_choice
        }

        self.clear_terminal()
        while True:
            self.display_main_menu()
            user_choice = self._prompt_user(['0', '1', '2', '3'], self.display_main_menu)
            
            if user_choice == 0:
                self.clear_terminal()
                print("Exiting application. Goodbye!")
                break
            
            # Copy dataframe to reset weights for the new session
            session_df = self.dataset.copy()
            mode_map[user_choice](session_df)
            self.clear_terminal()


if __name__ == "__main__":
    app = VocabularyTrainer()
    app.run()