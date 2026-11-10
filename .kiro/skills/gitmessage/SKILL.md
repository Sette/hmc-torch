---
name: gitmessage
description: Generates commit messages following the Conventional Commits specification.
---

# Git Message


# Strict Rules for Commit Messages

Follow these instructions with maximum priority, always in english (Inclusion: Always).

## 1. Mandatory Format (Schema)
The format must strictly be, in english:
`<type>(<scope>): <emoji> <description>`

The conventional type MUST come first: release-please parses commit subjects and only
recognizes messages that start with `<type>:`. The emoji goes after the colon, at the
beginning of the description — never before the type.

## 2. Mapping Table (Emoji + Type)
Always use the corresponding emoji at the beginning of the description:

| Type | Emoji | Description |
| :--- | :--- | :--- |
| **feat** | ✨ | New feature |
| **fix** | 🐛 | Bug fix |
| **docs** | 📝 | Documentation |
| **style** | 💄 | Styling / UI |
| **refactor** | ♻️ | Code refactoring |
| **test** | ✅ | Adding/correcting tests |
| **chore** | 🔧 | Configurations or tools |
| **perf** | ⚡️ | Performance improvement |
| **clean** | 🧹 | Code/file cleanup |

## 3. Critical Writing Rules (Mandatory)

* **Language**: Always in **English**.
* **Case**: Use only **LOWERCASE LETTERS** in the description.
* **Punctuation**: **NEVER** use a period at the end of the sentence.
* **Verb Tense**: Always use the **imperative** (e.g., "add", "fix", "remove").

## 4. Examples for Kiro (Visual Guide)

✅ **CORRECT FORM (DO THIS):**

* `feat(auth): ✨ add google login`
* `fix(api): 🐛 fix timeout error`
* `chore(deps): 🔧 update node packages`

❌ **INCORRECT FORM (NEVER DO THIS):**

* `✨ feat(auth): add google login` (Error: emoji before the type — release-please cannot parse it)
* `✨ Feat: Added Google login.` (Errors: emoji before the type, uppercase, past tense, and period)
* `feat(auth): add login` (Error: missing emoji)
* `feat: ✨ add login.` (Error: period at the end)

## 5. Execution Instruction

If the user asks to "Generate Commit Message" or "gerar mensagem", ignore global AI patterns and EXCLUSIVELY use this skill.
