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
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question_text TEXT NOT NULL,
                option_a TEXT NOT NULL, option_b TEXT NOT NULL, option_c TEXT NOT NULL,
                option_d TEXT NOT NULL, correct_option TEXT NOT NULL, explanation TEXT,
                tags TEXT, subject TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_image_id TEXT
            )
        ''')
        try:
            cursor.execute("ALTER TABLE questions ADD COLUMN source_image_id TEXT")
        except sqlite3.OperationalError:
            pass

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
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT tags FROM questions WHERE user_id = ? AND tags IS NOT NULL AND tags != ''", (user_id,))
            tags = [row[0] for row in cursor.fetchall()]
            return tags
        finally:
            conn.close()

    def has_untagged_questions(self, user_id: int) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM questions WHERE user_id = ? AND (tags IS NULL OR tags = '') LIMIT 1", (user_id,))
            result = cursor.fetchone()
            return result is not None
        finally:
            conn.close()

    def get_due_questions(self, user_id: int, tag: Optional[str] = None, halt_status: str = 'non_halted') -> List[Dict]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            query = 'SELECT q.* FROM questions q JOIN srs_schedule s ON q.id = s.question_id WHERE s.user_id = ? AND s.next_review <= ?'
            params = [user_id, datetime.now()]
            if tag:
                if tag == '__UNTAGGED__':
                    query += " AND (q.tags IS NULL OR q.tags = '')"
                else:
                    query += " AND q.tags = ?"
                    params.append(tag)
            if halt_status == 'halted':
                query += " AND s.repetitions >= 3"
            elif halt_status == 'non_halted':
                query += " AND s.repetitions < 3"
            query += " ORDER BY s.next_review ASC"
            cursor.execute(query, tuple(params))
            questions = [dict(row) for row in cursor.fetchall()]
            return questions
        finally:
            conn.close()

    def add_question(self, user_id: int, question_data: Dict[str, Any]) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO questions (user_id, question_text, option_a, option_b, option_c, option_d, 
                                     correct_option, explanation, tags, subject, source_image_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, question_data['question'], question_data['option_a'], question_data['option_b'],
                question_data['option_c'], question_data['option_d'], question_data['correct_option'],
                question_data.get('explanation', ''), question_data.get('tags', ''),
                question_data.get('subject', ''), question_data.get('source_image_id', None)
            ))
            question_id = cursor.lastrowid
            if question_id:
                cursor.execute('INSERT INTO srs_schedule (user_id, question_id, next_review) VALUES (?, ?, ?)',
                               (user_id, question_id, datetime.now()))
            conn.commit()
            return question_id
        finally:
            conn.close()
    
    def update_srs(self, user_id: int, question_id: int, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT interval_days, ease_factor, repetitions FROM srs_schedule WHERE user_id = ? AND question_id = ?', (user_id, question_id))
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
                new_repetitions = 0
                new_ease_factor = max(1.3, ease_factor - 0.2)
            next_review = datetime.now() + timedelta(days=new_interval)
            cursor.execute('''
                UPDATE srs_schedule SET interval_days = ?, ease_factor = ?, repetitions = ?, next_review = ?, last_answered = ?
                WHERE user_id = ? AND question_id = ?
            ''', (new_interval, new_ease_factor, new_repetitions, next_review, datetime.now(), user_id, question_id))
            conn.commit()
        finally:
            conn.close()
    
    def update_user_stats(self, user_id: int, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        try:
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
                    UPDATE user_stats SET total_questions = ?, correct_answers = ?, wrong_answers = ?, current_streak = ?, best_streak = ?, last_activity = ?
                    WHERE user_id = ?
                ''', (total, correct, wrong, current_streak, best_streak, datetime.now(), user_id))
            conn.commit()
        finally:
            conn.close()
    
    def get_user_stats(self, user_id: int) -> Dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row: return dict(row)
            return {'total_questions': 0, 'correct_answers': 0, 'wrong_answers': 0, 'current_streak': 0, 'best_streak': 0}
        finally:
            conn.close()

    def get_questions_by_tag(self, user_id: int, tag: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT id, question_text, option_a, option_b, option_c, option_d, correct_option, tags, subject FROM questions WHERE user_id = ? AND tags LIKE ?', (user_id, f'%{tag}%'))
            questions = [dict(row) for row in cursor.fetchall()]
            return questions
        finally:
            conn.close()
    
    def export_questions(self, user_id: int) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT question_text, option_a, option_b, option_c, option_d, correct_option, explanation, tags, subject, source_image_id FROM questions WHERE user_id = ?', (user_id,))
            questions = cursor.fetchall()
            if not questions: return ""
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option', 'Explanation', 'Tags', 'Subject', 'Source Image ID'])
            writer.writerows(questions)
            return output.getvalue()
        finally:
            conn.close()

    def import_questions(self, user_id: int, csv_data: str) -> Dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Clean the CSV data - remove BOM if present
            if csv_data.startswith('\ufeff'):
                csv_data = csv_data[1:]
            
            csv_reader = csv.DictReader(StringIO(csv_data))
            stats = {'imported': 0, 'skipped': 0, 'errors': 0}
            
            # Log the fieldnames for debugging
            logger.info(f"CSV fieldnames: {csv_reader.fieldnames}")
            
            field_mapping = {
                'Question': 'question', 
                'Option A': 'option_a', 
                'Option B': 'option_b', 
                'Option C': 'option_c', 
                'Option D': 'option_d', 
                'Correct Option': 'correct_option', 
                'Explanation': 'explanation', 
                'Tags': 'tags', 
                'Subject': 'subject', 
                'Source Image ID': 'source_image_id'
            }
            
            for row_num, row in enumerate(csv_reader, start=1):
                try:
                    logger.info(f"Processing row {row_num}: {dict(row)}")
                    
                    # Check essential fields
                    essential = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option']
                    missing_fields = [field for field in essential if not row.get(field, '').strip()]
                    
                    if missing_fields:
                        logger.warning(f"Row {row_num} skipped - missing fields: {missing_fields}")
                        stats['skipped'] += 1
                        continue
                    
                    # Map the fields
                    question_data = {}
                    for csv_field, db_field in field_mapping.items():
                        value = row.get(csv_field, '').strip()
                        question_data[db_field] = value
                    
                    # Validate correct option
                    correct_option = question_data['correct_option'].upper()
                    if correct_option not in ['A', 'B', 'C', 'D']:
                        logger.warning(f"Row {row_num} skipped - invalid correct option: {correct_option}")
                        stats['skipped'] += 1
                        continue
                    
                    question_data['correct_option'] = correct_option
                    
                    # Insert into database
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
                        question_data.get('source_image_id') or None
                    ))
                    
                    question_id = cursor.lastrowid
                    if question_id:
                        cursor.execute(
                            'INSERT INTO srs_schedule (user_id, question_id, next_review) VALUES (?, ?, ?)', 
                            (user_id, question_id, datetime.now())
                        )
                    
                    stats['imported'] += 1
                    logger.info(f"Successfully imported row {row_num}")
                    
                except Exception as e:
                    logger.error(f"Error importing row {row_num}: {row}. Error: {e}", exc_info=True)
                    stats['errors'] += 1
            
            conn.commit()
            stats['total'] = stats['imported'] + stats['skipped'] + stats['errors']
            logger.info(f"Import completed: {stats}")
            return stats
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Fatal CSV import error: {e}", exc_info=True)
            return {'imported': 0, 'skipped': 0, 'errors': 1, 'total': 0}
        finally:
            conn.close()
    
    def delete_question(self, question_id: int):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM answer_history WHERE question_id = ?", (question_id,))
            cursor.execute("DELETE FROM srs_schedule WHERE question_id = ?", (question_id,))
            cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
            conn.commit()
            logger.info(f"Deleted question with ID: {question_id}")
        finally:
            conn.close()

class ImageProcessor:
    """
    Analyzes images to extract educational content like MCQs, text, or labeled diagrams,
    with support for additional user prompts and option shuffling.
    """

    def _shuffle_options(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shuffles the options of an MCQ and updates the correct_option letter.
        """
        options = []
        for key in ['option_a', 'option_b', 'option_c', 'option_d', 'option_e']:
            if data.get(key):
                options.append(data[key])

        if not options or not data.get("correct_option"):
            return data

        correct_option_letter = data.get("correct_option", "A").upper()
        correct_option_index = ord(correct_option_letter) - ord('A')
        
        if correct_option_index < 0 or correct_option_index >= len(options):
            return data
            
        correct_answer_text = options[correct_option_index]
        random.shuffle(options)
        new_correct_letter = ""
        for i in range(5): data.pop(f"option_{chr(ord('a') + i)}", None)

        for i, option_text in enumerate(options):
            new_key = f"option_{chr(ord('a') + i)}"
            data[new_key] = option_text
            if option_text == correct_answer_text:
                new_correct_letter = chr(ord('A') + i)
        
        data["correct_option"] = new_correct_letter
        return data

    async def analyze_image(self, image_bytes: bytes, additional_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            prompt = f"""
You are an expert AI assistant for a medical student. Your task is to analyze an image, classify it, and return a precise JSON object based on the rules below.

**IMPORTANT USER INSTRUCTION:** You MUST consider this additional instruction if provided: '{additional_prompt if additional_prompt else "None"}'

First, classify the image into one of three types:
1. "MCQ_SCREENSHOT": A screenshot of a pre-existing Multiple Choice Question.
2. "INFORMATIONAL_TEXT": A page of text, notes, or a table for studying.
3. "LABELED_DIAGRAM": An anatomical or scientific image with parts labeled.

---
**IF "MCQ_SCREENSHOT", FOLLOW THESE RULES:**
1.  **Extract the MCQ:** Find the question, all options, and the correct answer indicated by a visual cue (green highlight, checkmark ✓, "Correct").
2.  **Return this exact JSON format:**
    {{
      "type": "mcq",
      "data": {{ "question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_option": "LETTER", "subject": "..." }}
    }}

---
**IF "INFORMATIONAL_TEXT", FOLLOW THESE RULES: (This is corrected)**
1.  **Extract All Text:** Your ONLY job is to extract all the relevant text from the image.
2.  **Return this exact JSON format:** This format signals that questions should be generated from the content.
    {{
      "type": "text",
      "data": {{ "content": "All the extracted text from the image." }}
    }}

---
**IF "LABELED_DIAGRAM", FOLLOW THESE RULES: (This is corrected)**
1.  **Describe the Image:** Your ONLY job is to describe the image and its labels in text form. For example: "A diagram of the heart showing the aorta, left ventricle, and right atrium."
2.  **Return this exact JSON format:** This format signals that questions should be generated from the content.
    {{
      "type": "text",
      "data": {{ "content": "A text description of the labeled diagram, including all the labels." }}
    }}
"""
            
            response = model.generate_content([prompt, image])
            
            response_text = response.text
            json_start_index = response_text.find('{')
            json_end_index = response_text.rfind('}')

            if json_start_index != -1 and json_end_index > json_start_index:
                json_str = response_text[json_start_index : json_end_index + 1]
                
                parsed_json = json.loads(json_str)

                if parsed_json.get("type") == "mcq" and parsed_json.get("data"):
                    parsed_json["data"] = self._shuffle_options(parsed_json["data"])
                
                return parsed_json
            else:
                return None

        except Exception as e:
            return None


class NEETPGBot:
    def __init__(self):
        self.db = DatabaseManager()
        self.processor = ImageProcessor()
        self.current_questions = {}
        self.GET_TEXT_COUNT, self.GET_IMAGE_COUNT, self.SELECT_REVIEW_TAG, self.SELECT_HALT_STATUS = range(4)
        self.WAITING_FOR_CSV = 101
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        context.user_data.clear()
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

    # THIS METHOD IS NOW CORRECTED
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("--- handle_photo triggered ---")
        if 'review_session' in context.user_data:
            del context.user_data['review_session']
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        caption = update.message.caption or ""
        
        # Extract hashtags for auto-tagging
        hashtags = [caption[entity.offset:entity.offset + entity.length] 
                    for entity in (update.message.caption_entities or []) 
                    if entity.type == MessageEntity.HASHTAG]
        tags = " ".join(hashtags)
        
        # --- START OF FIX ---
        # Create a clean version of the caption without hashtags to send to the AI
        clean_caption = caption
        for tag in hashtags:
            clean_caption = clean_caption.replace(tag, "")
        clean_caption = clean_caption.strip()
        # --- END OF FIX ---
        
        processing_msg = await update.message.reply_text("📸 Analyzing your image...")
        try:
            file = await context.bot.get_file(photo.file_id)
            file_bytes = await file.download_as_bytearray()
            
            # Pass the CLEAN caption as the additional_prompt
            analysis_result = await self.processor.analyze_image(bytes(file_bytes), additional_prompt=clean_caption)

            if not analysis_result or 'type' not in analysis_result or 'data' not in analysis_result:
                await processing_msg.edit_text("❌ Could not understand the image. Please try a clearer one.")
                return ConversationHandler.END

            if analysis_result['type'] == 'mcq':
                question_data = analysis_result['data']
                if tags:
                    question_data['tags'] = tags
                question_data['source_image_id'] = photo.file_id
                question_data['is_extracted_mcq'] = True
                question_id = self.db.add_question(user_id, question_data)
                await processing_msg.delete()
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

    async def send_quiz(self, chat, question_data: Dict[str, Any], question_id: int):
        if question_data.get('source_image_id') and not question_data.get('is_extracted_mcq', False):
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

    async def _send_next_review_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        review_session_questions = context.user_data.get('review_session', [])
        if review_session_questions:
            question_data = review_session_questions.pop(0)
            question_id = question_data['id']
            await self.send_quiz(update.effective_chat, question_data, question_id)
        else:
            await update.effective_chat.send_message("🎉 **Review session complete!** You've answered all due questions. Great job!")
            if 'review_session' in context.user_data:
                del context.user_data['review_session']
    
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
            context.user_data['selected_tag'] = tag
            keyboard = [
                [InlineKeyboardButton("▶️ Non-Halted Cards", callback_data=f"review_status:non_halted")],
                [InlineKeyboardButton("⏸️ Halted Cards (Reviewed 3+ times)", callback_data=f"review_status:halted")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(f"Reviewing **{tag_name}**. Which cards do you want to see?", reply_markup=reply_markup, parse_mode='Markdown')
            return self.SELECT_HALT_STATUS
        elif callback_data == "noop":
            return self.SELECT_REVIEW_TAG

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
        if 'review_session' in context.user_data:
            del context.user_data['review_session']
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
        else:
            if hasattr(context, 'user_data'):
                context.user_data.clear()
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
            logger.error(f"Export error: {e}", exc_info=True)
            await update.message.reply_text("❌ Export failed.")
    
    async def import_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle the /import command to start CSV import process"""
        await update.message.reply_text(
            "📥 **CSV Import**\n\n"
            "Please send your CSV file with the following columns:\n"
            "• Question\n"
            "• Option A, Option B, Option C, Option D\n"
            "• Correct Option\n"
            "• Explanation (optional)\n"
            "• Tags (optional)\n"
            "• Subject (optional)\n"
            "• Source Image ID (optional)\n\n"
            "Send your CSV file now:",
            parse_mode='Markdown'
        )
        return self.WAITING_FOR_CSV

    async def receive_csv_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # First thing: confirm the handler fired and what file arrived
        logger.info(
            "receive_csv_file triggered | user=%s | file=%s | mime=%s",
            update.effective_user.id,
            getattr(update.message.document, "file_name", None),
            getattr(update.message.document, "mime_type", None),
    )

        user_id = update.effective_user.id
        file_name = update.message.document.file_name
        processing_msg = await update.message.reply_text(f"Received {file_name}. Processing now...")

        try:
        # Download the CSV file content
            csv_file = await update.message.document.get_file()
            file_bytes = await csv_file.download_as_bytearray()
            csv_data = file_bytes.decode('utf-8-sig')

        # Optional: validate extension if you want to enforce .csv
        # if not (file_name or "").lower().endswith(".csv"):
        #     await processing_msg.edit_text("Please upload a .csv file exported from the bot.")
        #     return

        # Import questions using the database manager
            import_stats = self.db.import_questions(user_id, csv_data)

        # Create summary message
            summary = f"""
✅ **Import Complete!**

📊 **Results:**
• ✅ Imported: {import_stats['imported']}
• ⏭️ Skipped: {import_stats['skipped']}
• ❌ Errors: {import_stats['errors']}
• 📝 Total processed: {import_stats['total']}

All imported questions are scheduled for immediate review!
        """

            await processing_msg.edit_text(summary, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error processing CSV: {e}", exc_info=True)
            await processing_msg.edit_text(f"❌ An error occurred: {e}")

    
    async def import_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start CSV import and wait for the next uploaded file."""
        await update.message.reply_text(
        "📥 Please upload the CSV file now (the same format as /export). Send /cancel to abort."
        )
        return self.WAITING_FOR_CSV


    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.clear()
        await update.message.reply_text("Operation cancelled.")
        return ConversationHandler.END

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        stats = self.db.get_user_stats(user_id)

        msg = (
            "📊 **Your Study Stats:**\n"
            f"• Total questions answered: {stats['total_questions']}\n"
            f"• Correct answers: {stats['correct_answers']}\n"
            f"• Wrong answers: {stats['wrong_answers']}\n"
            f"• Current streak: {stats['current_streak']}\n"
            f"• Best streak: {stats['best_streak']}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")



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
    try:
        bot_instance = NEETPGBot()
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Start Flask thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

        # Command handlers
        application.add_handler(CommandHandler("start", bot_instance.start_command))
        application.add_handler(CommandHandler("stats", bot_instance.stats_command))
        application.add_handler(CommandHandler("export", bot_instance.export_command))
        application.add_handler(CommandHandler("import", bot_instance.import_command))

        # Import conversation: waits for CSV after /import
        import_conv_handler = ConversationHandler(
            entry_points=[CommandHandler("import", bot_instance.import_command)],
            states={
                bot_instance.WAITING_FOR_CSV: [
                    MessageHandler(filters.Document.ALL, bot_instance.receive_csv_file)
                ]
            },
            fallbacks=[CommandHandler("cancel", bot_instance.cancel)],
            per_message=False,
            allow_reentry=True,
        )
        application.add_handler(import_conv_handler)

        # Review conversation
        review_conv_handler = ConversationHandler(
            entry_points=[CommandHandler("review", bot_instance.review_command)],
            states={
                bot_instance.SELECT_REVIEW_TAG: [
                    CallbackQueryHandler(
                        bot_instance.handle_review_menu_callback,
                        pattern=r'^(review_all|select_tag|noop)'
                    )
                ],
                bot_instance.SELECT_HALT_STATUS: [
                    CallbackQueryHandler(
                        bot_instance.select_halt_status_callback,
                        pattern=r'^review_status:'
                    )
                ],
            },
            fallbacks=[CommandHandler('cancel', bot_instance.cancel)],
            per_message=False,
        )
        application.add_handler(review_conv_handler)

        # Unified text/photo conversation
        unified_conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.handle_text),
                MessageHandler(filters.PHOTO, bot_instance.handle_photo),
            ],
            states={
                bot_instance.GET_TEXT_COUNT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.receive_count_for_text)
                ],
                bot_instance.GET_IMAGE_COUNT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, bot_instance.receive_count_for_image)
                ],
            },
            fallbacks=[CommandHandler('cancel', bot_instance.cancel)],
        )
        application.add_handler(unified_conv_handler)

        # Other callback query handlers
        application.add_handler(CallbackQueryHandler(bot_instance.delete_callback, pattern=r'^delete_'))
        application.add_handler(CallbackQueryHandler(bot_instance.handle_answer, pattern=r'^answer_'))

        logger.info("🚀 NEET PG Bot started successfully!")
        application.run_polling()

    except Exception as e:
        logger.error(f"Failed to start bot: {e}")


if __name__ == '__main__':
    main()
