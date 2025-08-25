import pandas as pd
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
                tags TEXT, subject TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
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

    def get_due_questions(self, user_id: int, tag: Optional[str] = None) -> List[Dict]:
        """Gets due questions, with an optional filter for a specific tag."""
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
                                 correct_option, explanation, tags, subject)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            question_data.get('subject', '')
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
            new_repetitions = 0
            new_ease_factor = max(1.3, ease_factor - 0.8)
        
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
        
        cursor.execute('SELECT question_text, option_a, option_b, option_c, option_d, correct_option, explanation, tags, subject FROM questions WHERE user_id = ?', (user_id,))
        questions = cursor.fetchall()
        
        if not questions: return ""

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 
                        'Correct Option', 'Explanation', 'Tags', 'Subject'])
        
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
    async def analyze_image(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # MODIFIED: A rewritten, clearer prompt that correctly handles all cases.
            prompt = """
You are an expert AI assistant for a medical student. Your task is to analyze an image, classify it, and then follow the specific instructions for that type to return a precise JSON object.

First, classify the image into one of two types:
1.  "MCQ_SCREENSHOT": A screenshot of a pre-existing Multiple Choice Question.
2.  "INFORMATIONAL_TEXT": A page of text, notes, or a table for studying.

---
**IF THE IMAGE IS "MCQ_SCREENSHOT", FOLLOW THESE RULES:**
1.  **Find the Correct Answer:** You MUST determine the correct option by looking for visual indicators like a **green highlight, a green checkmark (✓), or the word "Correct"**. This is your top priority.
2.  **Extract the Full Question:** Combine the main question stem with any numbered lists or tables that are part of the question text itself.
3.  **Return this exact JSON format:**
{
  "type": "mcq",
  "data": {
    "question": "The full combined question text, including lists or tables.",
    "option_a": "Text for option A",
    "option_b": "Text for option B",
    "option_c": "Text for option C",
    "option_d": "Text for option D",
    "correct_option": "The letter (A, B, C, or D) identified from the highlight or checkmark",
    "subject": "Appropriate Medical Subject"
  }
}

---
**IF THE IMAGE IS "INFORMATIONAL_TEXT", FOLLOW THESE RULES:**
1.  **Prioritize Highlights:** Check for highlighted text first (e.g., with a yellow or green background). If found, you MUST extract ONLY the highlighted text.
2.  **Fallback to All Text:** If no text is highlighted, extract all the relevant text from the image as a fallback.
3.  **Return this exact JSON format:**
{
  "type": "text",
  "data": {
    "content": "The highlighted text ONLY, or all text as a fallback."
  }
}
"""
            response = model.generate_content([prompt, image])
            response_text = response.text.strip().replace("`", "")
            logger.info(f"AI Image Analysis Response: {response_text}")
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error: {e}\nRaw Response: {response_text}")
                return None

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return None


class NEETPGBot:

    def __init__(self):
        self.db = DatabaseManager()
        self.processor = ImageProcessor()
        self.current_questions = {}
        # MODIFIED: Add a state for selecting a review tag
        self.GET_TEXT_COUNT, self.GET_IMAGE_COUNT, self.SELECT_REVIEW_TAG = range(3)

        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = """
🩺 **Welcome to NEET PG MCQ Bot!**

**How it works:**
1. 📸 **Send any image:**
   - If it's an MCQ screenshot, I'll extract it.
   - If it's a page of text or a table, I'll extract the text and ask you how many questions to create from it.

2. 📝 **Send a topic** (e.g., "Anatomy of the heart"):
   - I'll ask how many questions you want and then generate them.

**Commands:**
📊 /stats  🔄 /review  📤 /export  ❌ /cancel
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    # NEW: This function now starts a conversation for photos
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # --- NEW: Diagnostic Logging ---
        logger.info("--- handle_photo triggered ---")
        logger.info(f"Message caption received: {update.message.caption}")
        logger.info(f"Caption entities received: {update.message.caption_entities}")
        # --- End of Diagnostic Logging ---

        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        
        caption = update.message.caption or ""
        entities = update.message.caption_entities or []
        hashtags = []
        for entity in entities:
            if entity.type == MessageEntity.HASHTAG:
                tag = caption[entity.offset : entity.offset + entity.length]
                hashtags.append(tag)
        tags = " ".join(hashtags)
        
        processing_msg = await update.message.reply_text("📸 Analyzing your image...")
        
        try:
            file = await context.bot.get_file(photo.file_id)
            file_bytes = await file.download_as_bytearray()
            
            analysis_result = await self.processor.analyze_image(bytes(file_bytes))

            if not analysis_result or 'type' not in analysis_result or 'data' not in analysis_result:
                await processing_msg.edit_text("❌ Could not understand the image. Please try a clearer one.")
                return ConversationHandler.END

            if analysis_result['type'] == 'mcq':
                question_data = analysis_result['data']
                
                if tags:
                    question_data['tags'] = tags
                
                question_id = self.db.add_question(user_id, question_data)
                await processing_msg.delete()
                await self.send_quiz(update, question_data, question_id)
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
            logger.error(f"Photo processing error: {e}")
            await processing_msg.edit_text(f"❌ **Processing Error**\n\nError: {str(e)}")
            return ConversationHandler.END



    async def receive_count_for_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        extracted_text = context.user_data.pop('image_text', None)
        # CORRECTED: This line now correctly retrieves the stored tag
        tags = context.user_data.pop('image_tags', "")

        if not extracted_text:
            await update.message.reply_text("🤔 Something went wrong. Please send the image again.")
            return ConversationHandler.END

        try:
            count = int(update.message.text)
            if not 1 <= count <= 10:
                await update.message.reply_text("⚠️ Please enter a number between 1 and 10.")
                context.user_data['image_text'] = extracted_text
                if tags: context.user_data['image_tags'] = tags # Put it back
                return self.GET_IMAGE_COUNT
        except (ValueError, TypeError):
            await update.message.reply_text("That doesn't look like a valid number. Please try again.")
            context.user_data['image_text'] = extracted_text
            if tags: context.user_data['image_tags'] = tags # Put it back
            return self.GET_IMAGE_COUNT

        # Pass the tags to the generation loop
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
                context.user_data['topic'] = topic # Put it back
                return self.GET_TEXT_COUNT
        except (ValueError, TypeError):
            await update.message.reply_text("That doesn't look like a valid number. Please try again.")
            context.user_data['topic'] = topic # Put it back
            return self.GET_TEXT_COUNT
        
        # We assume text-based topics don't have tags, so we pass an empty string
        return await self.generate_mcqs_loop(update, context, topic, count, tags="")


    
         # Centralized loop for generating questions, now with answer shuffling
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
                    prompt_addition = f"""
IMPORTANT: You have already generated the questions listed below. DO NOT repeat them or ask about the same specific details. Create a new, distinct question.

Previously Generated Questions:
{previous_qs}
"""
                
                prompt = f"""
Create a unique, high-quality NEET PG medical MCQ based on the following text: {source_text}.
{prompt_addition}

Return ONLY a single, valid JSON object, with the correct answer as option "A".
{{
  "question": "...", "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...", "correct_option": "A", "explanation": "...", "subject": "..."
}}
"""
                generation_config = {"temperature": 0.8}
                
                response = model.generate_content(prompt, generation_config=generation_config)
                response_text = response.text.strip().replace("```json", "").replace("```", "")
                
                logger.info(f"AI Raw Response #{i+1}: {response_text}")
                question_data = json.loads(response_text)
                
                generated_questions_text.append(question_data['question'])

                correct_answer_text = question_data['option_a']
                options = [
                    question_data['option_a'],
                    question_data['option_b'],
                    question_data['option_c'],
                    question_data['option_d']
                ]
                random.shuffle(options)

                question_data['option_a'] = options[0]
                question_data['option_b'] = options[1]
                question_data['option_c'] = options[2]
                question_data['option_d'] = options[3]

                new_correct_index = options.index(correct_answer_text)
                question_data['correct_option'] = ['A', 'B', 'C', 'D'][new_correct_index]
                
                # NEW: Add the tags from the caption if they exist.
                if tags:
                    question_data['tags'] = tags

                question_id = self.db.add_question(user_id, question_data)
                await self.send_quiz(update, question_data, question_id)
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

    async def import_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Asks the user to send a CSV file for importing."""
        await update.message.reply_text(
            "Ready to import! Please send your .csv file as a document.\n\n"
            "Make sure it has the columns: 'Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Option'."
        )

    async def receive_csv_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles receiving a CSV file for import."""
        # Note: Added 'self' since it's a method of the class
        file_name = update.message.document.file_name
        await update.message.reply_text(f"Received {file_name}. Processing now...")

        try:
            # Download the file
            csv_file = await update.message.document.get_file()
            file_path = await csv_file.download_to_drive()

            # Read the CSV using pandas
            df = pd.read_csv(file_path)

            # TODO: Add logic to loop through the dataframe (df)
            # and insert the questions into the database using self.db.add_question
            
            # For example:
            # user_id = update.effective_user.id
            # for index, row in df.iterrows():
            #     question_data = {
            #         'question': row['Question'],
            #         'option_a': row['Option A'],
            #         'option_b': row['Option B'],
            #         'option_c': row['Option C'],
            #         'option_d': row['Option D'],
            #         'correct_option': row['Correct Option'],
            #         'explanation': row.get('Explanation', ''),
            #         'tags': row.get('Tags', ''),
            #         'subject': row.get('Subject', '')
            #     }
            #     self.db.add_question(user_id, question_data)

            record_count = len(df)
            await update.message.reply_text(f"✅ Success! Imported {record_count} records from {file_name}.")

        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            await update.message.reply_text(f"❌ Sorry, an error occurred while processing the file: {e}")
    

   
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data.clear()
        await update.message.reply_text("Operation cancelled.")
        return ConversationHandler.END
    
    # ... (send_quiz, handle_answer, and other command handlers remain unchanged) ...
    async def send_quiz(self, update: Update, question_data: Dict[str, Any], question_id: int):
        # ADDED: A new "Delete" button
        keyboard = [
            [
                InlineKeyboardButton("A", callback_data=f"answer_A_{question_id}"),
                InlineKeyboardButton("B", callback_data=f"answer_B_{question_id}")
            ],
            [
                InlineKeyboardButton("C", callback_data=f"answer_C_{question_id}"),
                InlineKeyboardButton("D", callback_data=f"answer_D_{question_id}")
            ],
            [
                InlineKeyboardButton("Delete 🗑️", callback_data=f"delete_{question_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        q_text = question_data.get('question_text', question_data.get('question', ''))
        
        tags_line = ""
        if question_data.get('tags'):
            tags_line = f"\n🔖 **Tags:** {question_data['tags']}"

        quiz_text = f"""
❓ **Question:**
{q_text}

A) {question_data['option_a']}
B) {question_data['option_b']}
C) {question_data['option_c']}
D) {question_data['option_d']}

📚 **Subject:** {question_data.get('subject', 'N/A')}{tags_line}
        """
        
        self.current_questions[question_id] = question_data
        
        await update.message.reply_text(quiz_text, reply_markup=reply_markup)


    # This function's body is now correctly indented.
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
        cursor.execute('''
            INSERT INTO answer_history (user_id, question_id, user_answer, is_correct)
            VALUES (?, ?, ?, ?)
        ''', (user_id, question_id, user_answer, is_correct))
        conn.commit()
        conn.close()
        
        response = f"✅ **Correct!** Excellent work! 🎉\n\n" if is_correct else f"❌ **Wrong.** The correct answer was **{correct_option}**.\n\n"
        
        explanation = question_data.get('explanation', '')
        if explanation:
            response += f"💡 **Explanation:**\n{explanation}\n\n"
        
        response += "📅 Scheduled for review based on your answer."
        
        await query.edit_message_text(response, parse_mode='Markdown')
        await query.answer()

        # If we are in a review session, automatically send the next question.
        if 'review_session' in context.user_data:
            await self._send_next_review_question(update, context)

    # This function is now correctly un-indented to be at the same level as handle_answer.
    async def _send_next_review_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Helper function to send the next question in a review session."""
        review_session_questions = context.user_data.get('review_session', [])

        if review_session_questions:
            question_data = review_session_questions.pop(0)
            question_id = question_data['id']
            
            # ADDED: A new "Delete" button
            keyboard = [
                [
                    InlineKeyboardButton("A", callback_data=f"answer_A_{question_id}"),
                    InlineKeyboardButton("B", callback_data=f"answer_B_{question_id}")
                ],
                [
                    InlineKeyboardButton("C", callback_data=f"answer_C_{question_id}"),
                    InlineKeyboardButton("D", callback_data=f"answer_D_{question_id}")
                ],
                [
                    InlineKeyboardButton("Delete 🗑️", callback_data=f"delete_{question_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            q_text = question_data.get('question_text', question_data.get('question', ''))

            tags_line = ""
            if question_data.get('tags'):
                tags_line = f"\n🔖 **Tags:** {question_data['tags']}"

            quiz_text = f"""
❓ **Question:**
{q_text}

A) {question_data['option_a']}
B) {question_data['option_b']}
C) {question_data['option_c']}
D) {question_data['option_d']}

📚 **Subject:** {question_data.get('subject', 'N/A')}{tags_line}
            """
            self.current_questions[question_id] = question_data
            
            await update.effective_chat.send_message(quiz_text, reply_markup=reply_markup)
        else:
            await update.effective_chat.send_message("🎉 **Review session complete!** You've answered all due questions. Great job!")
            if 'review_session' in context.user_data:
                del context.user_data['review_session']

    async def delete_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles the 'Delete' button press."""
        query = update.callback_query
        
        # Extract the question ID from the callback data (e.g., "delete_123")
        question_id = int(query.data.split('_')[1])
        
        try:
            # Call the database function to delete the question
            self.db.delete_question(question_id)
            
            # Let the user know it was successful by editing the message
            await query.edit_message_text(f"🗑️ Question ID {question_id} has been permanently deleted.")
            
            # If the deleted question was part of a review, send the next one
            if 'review_session' in context.user_data:
                await self._send_next_review_question(update, context)

        except Exception as e:
            logger.error(f"Error deleting question {question_id}: {e}")
            await query.answer("❌ Error deleting question.", show_alert=True)
        else:
            # Acknowledge the button press
            await query.answer("Question deleted.")

    async def review_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Starts the review conversation by showing a menu of tags."""
        user_id = update.effective_user.id
        tags = self.db.get_unique_tags(user_id)
        has_untagged = self.db.has_untagged_questions(user_id)

        if not tags and not has_untagged:
            await update.message.reply_text("You have no questions to review yet. Add some first!")
            return ConversationHandler.END

        keyboard = []
        # Create a button for each tag
        for tag in tags:
            keyboard.append([InlineKeyboardButton(tag, callback_data=f"review_tag:{tag}")])
        
        # Add the "Untagged" button if needed
        if has_untagged:
            keyboard.append([InlineKeyboardButton("Untagged Questions", callback_data="review_tag:__UNTAGGED__")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Please choose a tag to review:", reply_markup=reply_markup)
        
        return self.SELECT_REVIEW_TAG

    async def select_review_tag_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handles the user's tag selection and starts the review session."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        # Extract tag from callback data (e.g., "review_tag:Cardiology")
        selected_tag = query.data.split(':', 1)[1]

        due_questions = self.db.get_due_questions(user_id, tag=selected_tag)

        if not due_questions:
            tag_name = "Untagged" if selected_tag == '__UNTAGGED__' else selected_tag
            await query.edit_message_text(f"🎉 No questions due for review in the '{tag_name}' tag. Great job!")
            return ConversationHandler.END

        # Start the session (same logic as before)
        context.user_data['review_session'] = due_questions
        count = len(due_questions)
        tag_name = "Untagged" if selected_tag == '__UNTAGGED__' else selected_tag
        await query.edit_message_text(f"You have **{count}** questions due for review in the '{tag_name}' tag. Let's begin! 🚀", parse_mode='Markdown')
        
        # Send the first question
        await self._send_next_review_question(update, context)

        return ConversationHandler.END

    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (This function remains unchanged) ...
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
        
        stats_text = f"""
📊 **Your Progress**

📈 **Performance:**
• Total: {total}
• Correct: {correct} 
• Wrong: {wrong}
• Accuracy: {accuracy:.1f}%

🔥 **Streaks:**
• Current: {current_streak}
• Best: {best_streak}

📅 **Due for review:** {due_count}
        """
        conn.close()
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def export_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # ... (This function remains unchanged) ...
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
        review_conv_handler = ConversationHandler(
            entry_points=[CommandHandler("review", bot_instance.review_command)],
            states={
                bot_instance.SELECT_REVIEW_TAG: [CallbackQueryHandler(bot_instance.select_review_tag_callback, pattern=r'^review_tag:')],
            },
            fallbacks=[CommandHandler('cancel', bot_instance.cancel)],
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


