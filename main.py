import pandas as pd
import io
import os
import json
import sqlite3
import asyncio
import logging
import csv
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
from io import StringIO
import random
from flask import Flask
import threading


# Fix for event loop issues in Replit
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Telegram Bot imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)
from telegram import MessageEntity

# Gemini AI imports
import google.generativeai as genai
from PIL import Image
import io

# Configure logging to write to both the console and a file
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler to print to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Handler to write to a file named 'bot.log'
file_handler = logging.FileHandler("bot.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


# Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings)


class DatabaseManager:
    def __init__(self, db_path='neet_pg_bot.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question_text TEXT NOT NULL,
                option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL,
                option_d TEXT NOT NULL, correct_option TEXT NOT NULL, explanation TEXT,
                tags TEXT, subject TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_image_id TEXT
            )
        ''')

        # This code adds the new column for storing images if it doesn't exist.
        # It will only run once and is safe to keep.
        try:
            cursor.execute("ALTER TABLE questions ADD COLUMN source_image_id TEXT")
            logger.info("Database updated: Added 'source_image_id' column to questions table.")
        except sqlite3.OperationalError:
            pass # Column already exists, do nothing.

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS srs_schedule (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question_id INTEGER, next_review TIMESTAMP, interval_days INTEGER DEFAULT 1, ease_factor REAL DEFAULT 2.5, repetitions INTEGER DEFAULT 0, last_answered TIMESTAMP, FOREIGN KEY (question_id) REFERENCES questions (id))
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (user_id INTEGER PRIMARY KEY, total_questions INTEGER DEFAULT 0, correct_answers INTEGER DEFAULT 0, wrong_answers INTEGER DEFAULT 0, current_streak INTEGER DEFAULT 0, best_streak INTEGER DEFAULT 0, last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answer_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question_id INTEGER, user_answer TEXT, is_correct BOOLEAN, answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (question_id) REFERENCES questions (id))
        ''')
        conn.commit()
        conn.close()

    def get_unique_tags(self, user_id: int) -> List[str]:
        """Gets a list of all unique tags for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT tags FROM questions WHERE user_id = ? AND tags IS NOT NULL AND tags != ''",
            (user_id,)
        )
        tags = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tags

    def has_untagged_questions(self, user_id: int) -> bool:
        """Checks if a user has any questions without a tag."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM questions WHERE user_id = ? AND (tags IS NULL OR tags = '') LIMIT 1",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def get_due_questions(self, user_id: int, tag: Optional[str] = None, halt_status: str = 'non_halted') -> List[Dict]:
        """Gets due questions, with filters for tags and halt status (reviewed 3+ times)."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT q.* FROM questions q
            JOIN srs_schedule s ON q.id = s.question_id
            WHERE s.user_id = ? AND s.next_review <= ?
        '''
        params = [user_id, datetime.now()]

        if tag:
            if tag == '__UNTAGGED__':
                query += " AND (q.tags IS NULL OR q.tags = '')"
            else:
                query += " AND q.tags = ?"
                params.append(tag)
        
        # Add logic for halt status based on repetitions
        if halt_status == 'halted':
            query += " AND s.repetitions >= 3"
        elif halt_status == 'non_halted':
            query += " AND s.repetitions < 3"
            
        query += " ORDER BY s.next_review ASC"

        cursor.execute(query, tuple(params))
        questions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return questions

    def add_question(self, user_id: int, question_data: Dict[str, Any]) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO questions (user_id, question_text, option_a, option_b, option_c, option_d, 
                                 correct_option, explanation, tags, subject, source_image_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            question_data['question'],
            question_data['option_a'],
            question_data['option_b'],
            question_data['option_c'],
            question_data['option_d'],
            question_data['correct_option'],
            question_data.get('explanation', ''),
            question_data.get('tags', ''),
            question_data.get('subject', ''),
            question_data.get('source_image_id', None) # Handles the new image ID
        ))
        
        question_id = cursor.lastrowid
        
        if question_id:
            cursor.execute('''
                INSERT INTO srs_schedule (user_id, question_id, next_review)
                VALUES (?, ?, ?)
            ''', (user_id, question_id, datetime.now()))
        
        conn.commit()
        conn.close()
        return question_id
    
    def update_srs(self, user_id: int, question_id: int, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT interval_days, ease_factor, repetitions FROM srs_schedule
            WHERE user_id = ? AND question_id = ?
        ''', (user_id, question_id))
        
        result = cursor.fetchone()
        if not result: return

        interval_days, ease_factor, repetitions = result
        
        if is_correct:
            if repetitions == 0: new_interval = 1
            elif repetitions == 1: new_interval = 6
            else: new_interval = int(interval_days * ease_factor)
            new_repetitions = repetitions + 1
            new_ease_factor = ease_factor + (0.1 - (5 - 3) * (0.08 + (5 - 3) * 0.02))
        else:
            new_interval = 1
            new_repetitions = 0 # Reset streak on wrong answer
            new_ease_factor = max(1.3, ease_factor - 0.2) # Less harsh penalty
        
        next_review = datetime.now() + timedelta(days=new_interval)
        
        cursor.execute('''
            UPDATE srs_schedule SET
                interval_days = ?, ease_factor = ?, repetitions = ?,
                next_review = ?, last_answered = ?
            WHERE user_id = ? AND question_id = ?
        ''', (new_interval, new_ease_factor, new_repetitions, next_review, 
              datetime.now(), user_id, question_id))
        
        conn.commit()
        conn.close()
    
    def update_user_stats(self, user_id: int, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
        stats = cursor.fetchone()
        
        if not stats:
            cursor.execute('''
                INSERT INTO user_stats (user_id, total_questions, correct_answers, wrong_answers, current_streak, best_streak)
                VALUES (?, 1, ?, ?, ?, ?)
            ''', (user_id, 1 if is_correct else 0, 0 if is_correct else 1, 1 if is_correct else 0, 1 if is_correct else 0))
        else:
            total = stats['total_questions'] + 1
            correct = stats['correct_answers'] + (1 if is_correct else 0)
            wrong = stats['wrong_answers'] + (0 if is_correct else 1)
            current_streak = stats['current_streak'] + 1 if is_correct else 0
            best_streak = max(stats['best_streak'], current_streak)
            
            cursor.execute('''
                UPDATE user_stats SET
                    total_questions = ?, correct_answers = ?, wrong_answers = ?,
                    current_streak = ?, best_streak = ?, last_activity = ?
                WHERE user_id = ?
            ''', (total, correct, wrong, current_streak, best_streak, datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    def get_questions_by_tag(self, user_id: int, tag: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, question_text, option_a, option_b, option_c, option_d, correct_option, tags, subject 
            FROM questions WHERE user_id = ? AND tags LIKE ?
        ''', (user_id, f'%{tag}%'))
        
        questions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return questions
    
    def export_questions(self, user_id: int) -> str:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT question_text, option_a, option_b, option_c, option_d, correct_option, explanation, tags, subject, source_image_id FROM questions WHERE user_id = ?', (user_id,))
        questions = cursor.fetchall()
        
        if not questions: return ""

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 
                        'Correct Option', 'Explanation', 'Tags', 'Subject', 'Source Image ID'])
        
        writer.writerows(questions)
        conn.close()
        return output.getvalue()

    def delete_question(self, question_id: int):
        """Permanently deletes a question and its related data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM answer_history WHERE question_id = ?", (question_id,))
        cursor.execute("DELETE FROM srs_schedule WHERE question_id = ?", (question_id,))
        cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted question with ID: {question_id}")



class ImageProcessor:
    """
    Analyzes images to extract educational content like MCQs, text, or labeled diagrams,
    with support for additional user prompts and option shuffling.
    """

    # --- NEW HELPER METHOD FOR SHUFFLING ---
    def _shuffle_options(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shuffles the options of an MCQ and updates the correct_option letter.
        """
        options = []
        # Gather all provided options into a list
        for key in ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']:
            if data.get(key):
                options.append(data[key])

        if not options or not data.get("correct_option"):
            return data # Cannot shuffle if options or correct key are missing

        # Identify the text of the correct answer before shuffling
        correct_option_letter = data.get("correct_option", "A").upper()
        correct_option_index = ord(correct_option_letter) - ord('A')
        
        if correct_option_index < 0 or correct_option_index >= len(options):
            # logger.warning(f"Correct option '{correct_option_letter}' is invalid for the number of options. Skipping shuffle.")
            return data
            
        correct_answer_text = options[correct_option_index]

        # Shuffle the list of options
        random.shuffle(options)

        # Re-assign the shuffled options and find the new correct letter
        new_correct_letter = ""
        # Clear old option keys before setting new ones
        for i in range(5): data.pop(f"option_{chr(ord('a') + i)}", None)

        for i, option_text in enumerate(options):
            new_key = f"option_{chr(ord('a') + i)}"
            data[new_key] = option_text
            if option_text == correct_answer_text:
                new_correct_letter = chr(ord('A') + i)
        
        data["correct_option"] = new_correct_letter
        # logger.info(f"Options shuffled. New correct option: {new_correct_letter}")
        return data

    # --- MODIFIED CORE METHOD ---
    async def analyze_image(self, image_bytes: bytes, additional_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # This new prompt is much more powerful and includes all your feature requests.
            prompt = f"""
You are an expert AI assistant for a medical student. Your primary task is to analyze an image and create a structured JSON output.

**IMPORTANT USER INSTRUCTION:** You MUST follow this additional instruction if provided: '{additional_prompt if additional_prompt else "None"}'

First, classify the image into one of three types:
1. "MCQ_SCREENSHOT": A screenshot of a pre-existing Multiple Choice Question.
2. "INFORMATIONAL_TEXT": A page of text, notes, or a table for studying.
3. "LABELED_DIAGRAM": An anatomical or scientific image with parts labeled (e.g., with arrows or lines).

---
**IF "MCQ_SCREENSHOT":**
1.  **Find Correct Answer:** Find the correct option via visual cues (green highlight, checkmark ✓, "Correct").
2.  **Return JSON:**
    {{
      "type": "mcq", "data": {{ "question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_option": "LETTER", "subject": "..." }}
    }}

---
**IF "INFORMATIONAL_TEXT":**
1.  **Generate a high-yield MCQ** based on the most important information in the text.
2.  **Prioritize User Prompt:** The generated question should be guided by the user's additional instruction.
3.  **Return JSON:**
    {{
      "type": "mcq", "data": {{ "question": "Generated question...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "The correct answer", "correct_option": "Correct letter", "subject": "..." }}
    }}

---
**IF "LABELED_DIAGRAM":**
1.  **Goal:** Create a "What is the indicated structure?" question.
2.  **Analyze:** Identify one labeled part. The question should describe the location of the label (e.g., "What structure is indicated by the arrow pointing to the superior pole of the kidney?").
3.  **Options:** The correct answer is the name of the labeled part. Other options should be plausible but incorrect structures from the same anatomical region.
4.  **Return JSON:**
    {{
      "type": "mcq", "data": {{ "question": "Question describing the label's location...", "option_a": "Plausible wrong answer", "option_b": "Correct answer", "option_c": "...", "option_d": "...", "correct_option": "B", "subject": "Anatomy" }}
    }}
"""
            response = model.generate_content([prompt, image])
            
            response_text = response.text
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}')

            if json_start_index != -1 and json_end_index > json_start_index:
                json_str = response_text[json_start_index : json_end_index + 1]
                
                parsed_json = json.loads(json_str)

                # Automatically shuffle options for any generated or extracted MCQ
                if parsed_json.get("type") == "mcq" and parsed_json.get("data"):
                    parsed_json["data"] = self._shuffle_options(parsed_json["data"])
                
                return parsed_json
            else:
                # logger.error(f"Could not find a complete JSON object in the response: {response_text}")
                return None

        except Exception as e:
            # logger.error(f"Image analysis error: {e}", exc_info=True)
            return None




class NEETPGBot:

    def __init__(self):
        self.db = DatabaseManager()
        self.processor = ImageProcessor()
        self.current_questions = {}
        # MODIFIED: Add a new state for the second level of the review menu
        self.GET_TEXT_COUNT, self.GET_IMAGE_COUNT, self.SELECT_REVIEW_TAG, self.SELECT_HALT_STATUS = range(4)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = """
🩺 **Welcome to NEET PG MCQ Bot!**

**How it works:**
1. 📸 **Send any image:**
   - Add a caption to give special instructions to the AI.
   - Use #hashtags in the caption to automatically tag the question.
   - The AI can now create questions from labeled diagrams!

2. 📝 **Send a topic** (e.g., "Anatomy of the heart"):
   - I'll ask how many questions you want and then generate them.

**Commands:**
📊 /stats  🔄 /review  📤 /export  📥 /import  ❌ /cancel
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')

    # MODIFIED: handle_photo now supports additional prompts and saves the image ID
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("--- handle_photo triggered ---")
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        
        # The caption now serves as the 'additional_prompt' for the AI
        caption = update.message.caption or ""
        
        # Extract hashtags for auto-tagging
        hashtags = [caption[entity.offset:entity.offset + entity.length] 
                    for entity in (update.message.caption_entities or []) 
                    if entity.type == MessageEntity.HASHTAG]
        tags = " ".join(hashtags)
        
        processing_msg = await update.message.reply_text("📸 Analyzing your image...")
        
        try:
            file = await context.bot.get_file(photo.file_id)
            file_bytes = await file.download_as_bytearray()
            
            # MODIFIED: Pass the caption as the additional_prompt
            analysis_result = await self.processor.analyze_image(bytes(file_bytes), additional_prompt=caption)

            if not analysis_result or 'type' not in analysis_result or 'data' not in analysis_result:
                await processing_msg.edit_text("❌ Could not understand the image. Please try a clearer one.")
                return ConversationHandler.END

            if analysis_result['type'] == 'mcq':
                question_data = analysis_result['data']
                
                if tags:
                    question_data['tags'] = tags
                
                # NEW: Save the Telegram file_id to associate the image with the question
                question_data['source_image_id'] = photo.file_id
                
                question_id = self.db.add_question(user_id, question_data)
                await processing_msg.delete()
                
                # Pass the full question data, including the new image ID
                await self.send_quiz(update.effective_chat, question_data, question_id)
                
                await update.message.reply_text(f"✅ **MCQ Extracted & Saved!**\n📚 Subject: {question_data.get('subject', 'N/A')}")
                return ConversationHandler.END

            elif analysis_result['type'] == 'text':
                extracted_text = analysis_result['data'].get('content', '')
                if not extracted_text.strip():
                    await processing_msg.edit_text("❌ I found the image type but could not extract any text.")
                    return ConversationHandler.END
                
                context.user_data['image_text'] = extracted_text
                if tags:
                    context.user_data['image_tags'] = tags
                
                await processing_msg.edit_text(f"✅ Text extracted successfully!\n\nHow many MCQs would you like to create from this text? (1-10)")
                return self.GET_IMAGE_COUNT

            else:
                await processing_msg.edit_text("❌ Unrecognized image type.")
                return ConversationHandler.END

        except Exception as e:
            logger.error(f"Photo processing error: {e}", exc_info=True)
            await processing_msg.edit_text(f"❌ **Processing Error**\n\nAn unexpected error occurred.")
            return ConversationHandler.END

    # MODIFIED: send_quiz can now send an associated image
    async def send_quiz(self, chat, question_data: Dict[str, Any], question_id: int):
        # NEW: Check if there's an image to send with the question
        if question_data.get('source_image_id'):
            await chat.send_photo(photo=question_data['source_image_id'])

        keyboard = [
            [InlineKeyboardButton("A", callback_data=f"answer_A_{question_id}"), InlineKeyboardButton("B", callback_data=f"answer_B_{question_id}")],
            [InlineKeyboardButton("C", callback_data=f"answer_C_{question_id}"), InlineKeyboardButton("D", callback_data=f"answer_D_{question_id}")],
            [InlineKeyboardButton("Delete 🗑️", callback_data=f"delete_{question_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        q_text = question_data.get('question_text', question_data.get('question', ''))
        tags_line = f"\n🔖 **Tags:** {question_data['tags']}" if question_data.get('tags') else ""

        quiz_text = f"""
❓ **Question:**
{q_text}

A) {question_data.get('option_a', 'N/A')}
B) {question_data.get('option_b', 'N/A')}
C) {question_data.get('option_c', 'N/A')}
D) {question_data.get('option_d', 'N/A')}

📚 **Subject:** {question_data.get('subject', 'N/A')}{tags_line}
        """
        self.current_questions[question_id] = question_data
        await chat.send_message(quiz_text, reply_markup=reply_markup)

    # MODIFIED: The helper function for review questions also sends images now
    async def _send_next_review_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        review_session_questions = context.user_data.get('review_session', [])

        if review_session_questions:
            question_data = review_session_questions.pop(0)
            question_id = question_data['id']
            # We use effective_chat to send messages, which works for both commands and callbacks
            await self.send_quiz(update.effective_chat, question_data, question_id)
        else:
            await update.effective_chat.send_message("🎉 **Review session complete!** You've answered all due questions. Great job!")
            if 'review_session' in context.user_data:
                del context.user_data['review_session']
    
    # --- NEW & REWRITTEN REVIEW COMMANDS ---

    # NEW: A helper function to start any review session to avoid repeating code
    async def _start_review_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE, questions: list, description: str):
        query = update.callback_query
        if not questions:
            await query.edit_message_text(f"🎉 No questions are due for review in **{description}**. Great job!")
            return ConversationHandler.END
        
        context.user_data['review_session'] = questions
        count = len(questions)
        await query.edit_message_text(f"🚀 Starting review for **{description}**. You have **{count}** questions due. Let's begin!", parse_mode='Markdown')
        await self._send_next_review_question(update, context)
        return ConversationHandler.END

    # MODIFIED: review_command now shows the new top-level menu
    async def review_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = update.effective_user.id
        tags = self.db.get_unique_tags(user_id)
        has_untagged = self.db.has_untagged_questions(user_id)

        keyboard = [
            [InlineKeyboardButton("🚀 Review All Non-Halted Cards", callback_data="review_all:non_halted")],
            [InlineKeyboardButton("📚 Review All Halted Cards", callback_data="review_all:halted")]
        ]
        
        if tags or has_untagged:
            keyboard.append([InlineKeyboardButton("--- OR CHOOSE A TAG ---", callback_data="noop")])

        for tag in tags:
            keyboard.append([InlineKeyboardButton(f"🔖 {tag}", callback_data=f"select_tag:{tag}")])
        
        if has_untagged:
            keyboard.append([InlineKeyboardButton("📄 Untagged Questions", callback_data="select_tag:__UNTAGGED__")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Please choose a review category:", reply_markup=reply_markup)
        
        return self.SELECT_REVIEW_TAG

    # MODIFIED: This function now acts as a router for the main review menu
    async def handle_review_menu_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        callback_data = query.data

        if callback_data.startswith("review_all:"):
            halt_status = callback_data.split(':')[1]
            questions = self.db.get_due_questions(user_id, tag=None, halt_status=halt_status)
            description = f"All {halt_status.replace('_', ' ')} cards"
            return await self._start_review_session(update, context, questions, description)

        elif callback_data.startswith("select_tag:"):
            tag = callback_data.split(':', 1)[1]
            tag_name = "Untagged" if tag == '__UNTAGGED__' else tag
            
            context.user_data['selected_tag'] = tag # Store the tag for the next step

            keyboard = [
                [InlineKeyboardButton("▶️ Non-Halted Cards", callback_data=f"review_status:non_halted")],
                [InlineKeyboardButton("⏸️ Halted Cards (Reviewed 3+ times)", callback_data=f"review_status:halted")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(f"Reviewing **{tag_name}**. Which cards do you want to see?", reply_markup=reply_markup, parse_mode='Markdown')
            
            return self.SELECT_HALT_STATUS # Move to the next state
        
        elif callback_data == "noop":
            return self.SELECT_REVIEW_TAG # Do nothing, wait for a real choice

    # NEW: This function handles the second menu (choosing halt status for a specific tag)
    async def select_halt_status_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        halt_status = query.data.split(':')[1]
        tag = context.user_data.pop('selected_tag', None)

        if not tag:
            await query.edit_message_text("❌ An error occurred. Please start the review again with /review.")
            return ConversationHandler.END

        questions = self.db.get_due_questions(user_id, tag=tag, halt_status=halt_status)
        tag_name = "Untagged" if tag == '__UNTAGGED__' else tag
        description = f"{halt_status.replace('_', ' ')} cards in '{tag_name}'"
        return await self._start_review_session(update, context, questions, description)
        
    # --- UNCHANGED EXISTING METHODS ---

    async def receive_count_for_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        extracted_text = context.user_data.pop('image_text', None)
        tags = context.user_data.pop('image_tags', "")

        if not extracted_text:
            await update.message.reply_text("🤔 Something went wrong. Please send the image again.")
            return ConversationHandler.END

        try:
            count = int(update.message.text)
            if not 1 <= count <= 10:
                await update.message.reply_text("⚠️ Please enter a number between 1 and 10.")
                context.user_data['image_text'] = extracted_text
                if tags: context.user_data['image_tags'] = tags
                return self.GET_IMAGE_COUNT
        except (ValueError, TypeError):
            await update.message.reply_text("That doesn't look like a valid number. Please try again.")
            context.user_data['image_text'] = extracted_text
            if tags: context.user_data['image_tags'] = tags
            return self.GET_IMAGE_COUNT

        return await self.generate_mcqs_loop(update, context, extracted_text, count, tags=tags)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        topic = update.message.text
        context.user_data['topic'] = topic
        await update.message.reply_text(f"How many MCQs would you like to generate for the topic \"{topic}\"?\n\nPlease send a number (1-10).")
        return self.GET_TEXT_COUNT

    async def receive_count_for_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        topic = context.user_data.pop('topic', None)
        if not topic:
            await update.message.reply_text("🤔 Something went wrong. Please send the topic again.")
            return ConversationHandler.END
        
        try:
            count = int(update.message.text)
            if not 1 <= count <= 10:
                await update.message.reply_text("⚠️ Please enter a number between 1 and 10.")
                context.user_data['topic'] = topic
                return self.GET_TEXT_COUNT
        except (ValueError, TypeError):
            await update.message.reply_text("That doesn't look like a valid number. Please try again.")
            context.user_data['topic'] = topic
            return self.GET_TEXT_COUNT
        
        return await self.generate_mcqs_loop(update, context, topic, count, tags="")

    async def generate_mcqs_loop(self, update: Update, context: ContextTypes.DEFAULT_TYPE, source_text: str, count: int, tags: str = "") -> int:
        user_id = update.effective_user.id
        await update.message.reply_text(f"⚡ Creating {count} distinct MCQ(s) for you. This may take a moment...")
        
        questions_generated = 0
        generated_questions_text = []

        for i in range(count):
            try:
                prompt_addition = ""
                if generated_questions_text:
                    previous_qs = "\n".join(f"- \"{q}\"" for q in generated_questions_text)
                    prompt_addition = f"IMPORTANT: You have already generated the questions listed below. DO NOT repeat them... Previously Generated Questions:\n{previous_qs}"
                
                prompt = f"""Create a unique, high-quality NEET PG medical MCQ based on the following text: {source_text}. {prompt_addition} Return ONLY a single, valid JSON object, with the correct answer as option "A". {{"question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_option": "A", "explanation": "...", "subject": "..."}}"""
                generation_config = {"temperature": 0.8}
                
                response = model.generate_content(prompt, generation_config=generation_config)
                response_text = response.text.strip().replace("```json", "").replace("```", "")
                
                logger.info(f"AI Raw Response #{i+1}: {response_text}")
                question_data = json.loads(response_text)
                
                generated_questions_text.append(question_data['question'])
                correct_answer_text = question_data['option_a']
                options = [question_data['option_a'], question_data['option_b'], question_data['option_c'], question_data['option_d']]
                random.shuffle(options)

                question_data['option_a'] = options[0]
                question_data['option_b'] = options[1]
                question_data['option_c'] = options[2]
                question_data['option_d'] = options[3]

                new_correct_index = options.index(correct_answer_text)
                question_data['correct_option'] = ['A', 'B', 'C', 'D'][new_correct_index]
                
                if tags:
                    question_data['tags'] = tags

                question_id = self.db.add_question(user_id, question_data)
                await self.send_quiz(update.effective_chat, question_data, question_id)
                questions_generated += 1
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Text processing error on iteration {i+1}: {e}")
                await update.message.reply_text(f"⚠️ Could not generate question #{i+1}. Moving to the next...")
                continue
        
        if questions_generated > 0:
            await update.message.reply_text(f"✅ Finished! Generated {questions_generated} out of {count} requested questions.")
        else:
            await update.message.reply_text("❌ Failed to generate any questions from the provided source.")
        
        return ConversationHandler.END

    async def handle_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        user_id = query.from_user.id
        
        parts = query.data.split('_')
        user_answer = parts[1]
        question_id = int(parts[2])
        
        if question_id not in self.current_questions:
            await query.answer("❌ This question seems to have expired.", show_alert=True)
            await query.edit_message_text("This quiz has expired.")
            return
        
        question_data = self.current_questions.pop(question_id)
        correct_option = question_data['correct_option']
        is_correct = (user_answer == correct_option)
        
        self.db.update_srs(user_id, question_id, is_correct)
        self.db.update_user_stats(user_id, is_correct)
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO answer_history (user_id, question_id, user_answer, is_correct) VALUES (?, ?, ?, ?)', (user_id, question_id, user_answer, is_correct))
        conn.commit()
        conn.close()
        
        response = f"✅ **Correct!** Excellent work! 🎉\n\n" if is_correct else f"❌ **Wrong.** The correct answer was **{correct_option}**.\n\n"
        
        explanation = question_data.get('explanation', '')
        if explanation:
            response += f"💡 **Explanation:**\n{explanation}\n\n"
        
        response += "📅 Scheduled for review based on your answer."
        
        await query.edit_message_text(response, parse_mode='Markdown')
        await query.answer()

        if 'review_session' in context.user_data:
            await self._send_next_review_question(update, context)

    async def delete_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        question_id = int(query.data.split('_')[1])
        try:
            self.db.delete_question(question_id)
            await query.edit_message_text(f"🗑️ Question ID {question_id} has been permanently deleted.")
            if 'review_session' in context.user_data:
                await self._send_next_review_question(update, context)
        except Exception as e:
            logger.error(f"Error deleting question {question_id}: {e}")
            await query.answer("❌ Error deleting question.", show_alert=True)
        else:
            await query.answer("Question deleted.")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
        stats = cursor.fetchone()
        
        if not stats:
            await update.message.reply_text("📊 No stats yet. Answer some questions first!")
            conn.close()
            return
        
        total = stats['total_questions']
        correct = stats['correct_answers']
        wrong = stats['wrong_answers']
        current_streak = stats['current_streak']
        best_streak = stats['best_streak']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        cursor.execute('SELECT COUNT(*) FROM srs_schedule WHERE user_id = ? AND next_review <= ?', (user_id, datetime.now()))
        due_count = cursor.fetchone()[0]
        
        stats_text = f"📊 **Your Progress**\n\n📈 **Performance:**\n• Total: {total}\n• Correct: {correct} \n• Wrong: {wrong}\n• Accuracy: {accuracy:.1f}%\n\n🔥 **Streaks:**\n• Current: {current_streak}\n• Best: {best_streak}\n\n📅 **Due for review:** {due_count}"
        conn.close()
        await update.message.reply_text(stats_text, parse_mode='Markdown')

    async def export_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        try:
            csv_data = self.db.export_questions(user_id)
            if not csv_data.strip():
                await update.message.reply_text("📭 No questions to export yet!")
                return
            
            filename = f"neet_pg_export_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            await update.message.reply_document(
                document=io.BytesIO(csv_data.encode('utf-8')),
                filename=filename,
                caption="📤 Here are your exported questions!"
            )
        except Exception as e:
            logger.error(f"Export error: {e}")
            await update.message.reply_text("❌ Export failed.")
    
    async def import_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Ready to import! Please send your .csv file as a document.\n\nMake sure it has the columns: 'Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option'.")

    async def receive_csv_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        file_name = update.message.document.file_name
        await update.message.reply_text(f"Received {file_name}. Processing now...")
        try:
            csv_file = await update.message.document.get_file()
            file_path = await csv_file.download_to_drive()
            df = pd.read_csv(file_path)
            record_count = len(df)
            await update.message.reply_text(f"✅ TEST SUCCESS! File is readable and has {record_count} records.")
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            await update.message.reply_text(f"❌ An error occurred: {e}")
            
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.clear()
        await update.message.reply_text("Operation cancelled.")
        return ConversationHandler.END



app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive and running!"

def run_flask():
    # This line tells Flask to use the port Render provides, or 8080 as a backup.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

def main():
    """Starts and runs the bot."""
    # The 'try' block groups the main bot logic for error handling.
    try:
        bot_instance = NEETPGBot()
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Start the Flask web server in a separate thread to keep the bot alive on Render.
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Handler for conversations started by sending text or an image.
        unified_conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text),
                MessageHandler(filters.PHOTO, bot_instance.handle_photo)
            ],
            states={
                bot_instance.GET_TEXT_COUNT: [MessageHandler(filters.Regex(r'^\d+$'), bot_instance.receive_count_for_text)],
                bot_instance.GET_IMAGE_COUNT: [MessageHandler(filters.Regex(r'^\d+$'), bot_instance.receive_count_for_image)],
            },
            fallbacks=[CommandHandler('cancel', bot_instance.cancel)],
        )

        # A separate handler for the /review command conversation.
        # A separate handler for the /review command conversation.
    review_conv_handler = ConversationHandler(
    entry_points=[CommandHandler("review", bot_instance.review_command)],
    states={
        bot_instance.SELECT_REVIEW_TAG: [
            CallbackQueryHandler(bot_instance.handle_review_menu_callback)
        ],
        bot_instance.SELECT_HALT_STATUS: [
            CallbackQueryHandler(bot_instance.select_halt_status_callback)
        ],
    },
    fallbacks=[CommandHandler('cancel', bot_instance.cancel)],
    per_message=False
)

        
        # Add the conversation handlers to the application.
        application.add_handler(unified_conv_handler)
        application.add_handler(review_conv_handler)
        
        # Add all other standard command and callback handlers.
        # All of these must be indented to the same level.
        application.add_handler(CommandHandler("start", bot_instance.start_command))
        application.add_handler(CommandHandler("stats", bot_instance.stats_command))
        application.add_handler(CommandHandler("export", bot_instance.export_command))
        application.add_handler(CommandHandler("import", bot_instance.import_command))
        application.add_handler(MessageHandler(filters.Document.MimeType("text/csv"), bot_instance.receive_csv_file))
        application.add_handler(CallbackQueryHandler(bot_instance.delete_callback, pattern=r'^delete_'))
        application.add_handler(CallbackQueryHandler(bot_instance.handle_answer, pattern=r'^answer_'))
        
        logger.info("🚀 NEET PG Bot started successfully!")
        
        # Start polling for updates from Telegram.
        application.run_polling()
        
    # This 'except' block must be aligned with the 'try' statement.
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")

# This ensures the 'main' function is called only when the script is executed directly.
if __name__ == '__main__':
    main()


