from datetime import datetime, timedelta, timezone
from colorama import Fore
import os

from personality.bot_info import textmodel

def summarize_memory(self):
    """
    Summarize complete weeks of conversation history into weekly summaries.
    Only complete weeks (Sunday to Saturday) are summarized.
    """
    now = datetime.now(timezone.utc)
    text_llm_model: str = os.getenv("TEXT_LLM_MODEL", textmodel)

    # Parse and sort entries by timestamp
    entries = []
    for e in self.memory:
        raw_ts = e.get('timestamp')
        if not raw_ts:
            continue
            
        try:
            ts = self._parse_human_datetime(raw_ts)
            entries.append((ts, e))
        except Exception as ex:
            print(f"[Warning] Could not parse timestamp '{raw_ts}': {ex}")
            continue
    
    if not entries:
        print(Fore.CYAN + "No entries to summarize." + Fore.RESET)
        return
    
    entries.sort(key=lambda x: x[0])
    print(Fore.CYAN + f"Summarizing memory from {len(entries)} entries..." + Fore.RESET)
    
    # Find the first Sunday after the oldest entry
    oldest = entries[0][0]
    days_to_sunday = (6 - oldest.weekday()) % 7
    if days_to_sunday == 0 and oldest.weekday() == 6:  # Already Sunday
        week_start = oldest.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        if days_to_sunday == 0:  # Handle Monday case
            days_to_sunday = 7
        week_start = (oldest + timedelta(days=days_to_sunday)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    summaries = []
    used_entries = set()
    
    # Process complete weeks
    while True:
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        if week_end > now:
            break
        week_entries = [entry for ts, entry in entries if week_start <= ts <= week_end]
        
        if week_entries:
            # Build conversation text for this week
            conversation_parts = []
            user_entries = [e for e in week_entries if e.get('role') == 'user']
            assistant_entries = [e for e in week_entries if e.get('role') == 'assistant']
            
            # Pair up user and assistant entries
            for i in range(min(len(user_entries), len(assistant_entries))):
                user_entry = user_entries[i]
                assistant_entry = assistant_entries[i]
                
                timestamp = user_entry.get('timestamp', '')
                conversation_parts.append(f"[{timestamp}]")
                conversation_parts.append(f"{self.username}: {user_entry['content']}")
                conversation_parts.append(f"{self.botname}: {assistant_entry['content']}")
                conversation_parts.append("")  # Add spacing
            
            if conversation_parts:
                conversation_text = '\n'.join(conversation_parts)
                
                # Generate summary with human-readable dates
                week_start_str = week_start.strftime("%A, %B %d, %Y")
                week_end_str = week_end.strftime("%A, %B %d, %Y")
                
                prompt = (f"Summarize these interactions from {week_start_str} to {week_end_str} "
                        f"as a diary entry from the AI assistant {self.botname}'s perspective. Focus on key topics, emotions, and memorable moments. "
                        f"Summarize in one paragraph only. Do not create a dialog. Do not quote text from the conversation. Write the summary as a recollection "
                         "of events, interactions, and details from the assistant's perspective. The following is the conversation to summarize:\n\n"
                        f"{conversation_text}")
                print(Fore.CYAN + f"{prompt}" + Fore.RESET)
                try:
                    summary = self._call_ollama(prompt, text_llm_model)

                    summary_accepted = False
                    while not summary_accepted:
                        if input(Fore.CYAN + "Is this summary acceptable? (y/n)" + Fore.RESET) != "y":
                            print(Fore.RED + "Discarded. Running new summarization..." + Fore.RESET)
                            summary = self._call_ollama(prompt, text_llm_model)
                        else:
                            summary_accepted = True
            
                    if summary.strip():  # Only add non-empty summaries
                        summary_text = f"Weekly Summary ({week_start_str} to {week_end_str}): {summary.strip()}"
                        
                        # Add to embeddings
                        self._add_embedding(summary_text, {
                            'type': 'weekly_summary',
                            'week_start': week_start_str,
                            'week_end': week_end_str
                        })
                        
                        summaries.append({
                            'type': 'weekly_summary',
                            'week_start': week_start_str,
                            'week_end': week_end_str,
                            'summary': summary.strip(),
                            'timestamp': week_end.strftime("%A, %B %d, %Y at %I:%M %p UTC")
                        })
                        print(Fore.YELLOW + f"Created weekly summary for {week_start_str} to {week_end_str}" + Fore.RESET)
                        
                        # Mark entries as used
                        for ts, entry in entries:
                            if week_start <= ts <= week_end:
                                used_entries.add(id(entry))
                except Exception as e:
                    print(f"[Error] Failed to generate summary for week {week_start_str}: {e}")
        
        # Move to next week
        week_start += timedelta(days=7)
    
    # Keep unsummarized entries
    remaining_entries = [entry for ts, entry in entries if id(entry) not in used_entries]
    self.memory = remaining_entries
    self._save_memory()
    print(Fore.GREEN + f"Memory summarization complete. "
                      f"Created {len(summaries)} weekly summaries, "
                      f"kept {len(remaining_entries)} recent entries." + Fore.RESET)