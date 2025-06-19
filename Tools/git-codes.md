# Git Codes & Commands

A comprehensive collection of Git commands, workflows, and best practices for developers. From basic operations to advanced techniques, this guide covers everything you need to master Git.

## Table of Contents
- [Git Basics](#git-basics)
- [Configuration](#configuration)
- [Repository Management](#repository-management)
- [Branching & Merging](#branching--merging)
- [Stashing & Cleaning](#stashing--cleaning)
- [Remote Operations](#remote-operations)
- [Advanced Commands](#advanced-commands)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Git Workflows](#git-workflows)
- [Tools & Extensions](#tools--extensions)

## Git Basics

### Initial Setup
```bash
# Initialize a new repository
git init

# Clone an existing repository
git clone <repository-url>
git clone --depth 1 <repository-url>  # Shallow clone

# Check repository status
git status
git status -s  # Short format
```

### Basic Operations
```bash
# Add files to staging
git add <file>
git add .  # Add all files
git add -p  # Interactive staging

# Commit changes
git commit -m "Commit message"
git commit -am "Add and commit"  # Stage and commit tracked files
git commit --amend  # Modify last commit

# View history
git log
git log --oneline
git log --graph --oneline --all
git log -p  # Show patches
git log --follow <file>  # Follow file history through renames
```

## Configuration

### User Configuration
```bash
# Set user information
git config --global user.name "Your Name"
git config --global user.email "email@example.com"

# Set default editor
git config --global core.editor "vim"

# Set default branch name
git config --global init.defaultBranch main

# List all configurations
git config --list
git config --global --list
```

### Aliases
```bash
# Create useful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

### SSH Configuration
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "email@example.com"

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Test SSH connection
ssh -T git@github.com
```

## Repository Management

### Working with Files
```bash
# Remove files
git rm <file>
git rm --cached <file>  # Remove from index only

# Move/rename files
git mv <old-name> <new-name>

# View file changes
git diff  # Working directory vs staging
git diff --staged  # Staging vs last commit
git diff HEAD  # Working directory vs last commit
git diff <branch1>..<branch2>  # Between branches
```

### Undoing Changes
```bash
# Discard working directory changes
git checkout -- <file>
git restore <file>  # New syntax

# Unstage files
git reset HEAD <file>
git restore --staged <file>  # New syntax

# Reset commits
git reset --soft HEAD~1  # Keep changes staged
git reset --mixed HEAD~1  # Keep changes unstaged (default)
git reset --hard HEAD~1  # Discard changes

# Revert commits (creates new commit)
git revert <commit-hash>
git revert HEAD
```

## Branching & Merging

### Branch Operations
```bash
# List branches
git branch  # Local branches
git branch -r  # Remote branches
git branch -a  # All branches
git branch -v  # Verbose with last commit

# Create branch
git branch <branch-name>
git checkout -b <branch-name>  # Create and switch
git switch -c <branch-name>  # New syntax

# Switch branches
git checkout <branch-name>
git switch <branch-name>  # New syntax

# Delete branch
git branch -d <branch-name>  # Safe delete
git branch -D <branch-name>  # Force delete

# Rename branch
git branch -m <old-name> <new-name>
git branch -m <new-name>  # Rename current branch
```

### Merging
```bash
# Merge branch
git merge <branch-name>
git merge --no-ff <branch-name>  # No fast-forward

# Abort merge
git merge --abort

# Resolve conflicts
git add <resolved-file>
git merge --continue

# Merge strategies
git merge -s recursive <branch>  # Default
git merge -s ours <branch>  # Keep our version
git merge -s theirs <branch>  # Not available, use recursive -X theirs
```

### Rebasing
```bash
# Basic rebase
git rebase <branch>
git rebase -i HEAD~3  # Interactive rebase last 3 commits

# Continue/abort rebase
git rebase --continue
git rebase --abort

# Pull with rebase
git pull --rebase
git config pull.rebase true  # Set as default
```

## Stashing & Cleaning

### Stash Operations
```bash
# Save stash
git stash
git stash push -m "Message"
git stash --include-untracked

# List stashes
git stash list

# Apply stash
git stash apply  # Apply most recent
git stash apply stash@{n}  # Apply specific stash
git stash pop  # Apply and remove

# Remove stash
git stash drop stash@{n}
git stash clear  # Remove all stashes

# Create branch from stash
git stash branch <branch-name>
```

### Cleaning
```bash
# Remove untracked files
git clean -n  # Dry run
git clean -f  # Force remove files
git clean -fd  # Remove files and directories
git clean -fx  # Remove ignored files too
```

## Remote Operations

### Remote Management
```bash
# List remotes
git remote
git remote -v  # Verbose with URLs

# Add remote
git remote add <name> <url>
git remote add origin git@github.com:user/repo.git

# Remove/rename remote
git remote remove <name>
git remote rename <old> <new>

# Update remote URL
git remote set-url origin <new-url>
```

### Fetching & Pulling
```bash
# Fetch from remote
git fetch
git fetch <remote>
git fetch --all
git fetch --prune  # Remove deleted remote branches

# Pull changes
git pull
git pull <remote> <branch>
git pull --rebase
```

### Pushing
```bash
# Push to remote
git push
git push <remote> <branch>
git push -u origin main  # Set upstream
git push --force  # Force push (careful!)
git push --force-with-lease  # Safer force push

# Push tags
git push --tags
git push <remote> <tag>

# Delete remote branch
git push <remote> --delete <branch>
git push <remote> :<branch>  # Old syntax
```

## Advanced Commands

### Cherry-pick
```bash
# Apply specific commits
git cherry-pick <commit-hash>
git cherry-pick <hash1> <hash2>
git cherry-pick <hash1>..<hash2>  # Range (excludes hash1)

# Cherry-pick options
git cherry-pick -n <hash>  # No commit
git cherry-pick -x <hash>  # Add source reference
```

### Bisect (Binary Search)
```bash
# Find bug introduction
git bisect start
git bisect bad  # Current commit is bad
git bisect good <commit>  # Known good commit

# Mark commits
git bisect good
git bisect bad
git bisect skip

# End bisect
git bisect reset
```

### Reflog
```bash
# View reference log
git reflog
git reflog show <branch>

# Recover lost commits
git checkout <reflog-hash>
git reset --hard <reflog-hash>
```

### Submodules
```bash
# Add submodule
git submodule add <repository-url> <path>

# Initialize submodules
git submodule init
git submodule update
git submodule update --init --recursive

# Update submodules
git submodule update --remote
git submodule foreach git pull origin main
```

### Worktree
```bash
# Add worktree
git worktree add <path> <branch>
git worktree add -b <new-branch> <path>

# List worktrees
git worktree list

# Remove worktree
git worktree remove <path>
git worktree prune
```

## Troubleshooting

### Common Fixes
```bash
# Fix corrupted index
rm .git/index
git reset

# Recover deleted branch
git reflog
git checkout -b <branch-name> <hash>

# Fix "diverged" branches
git fetch origin
git reset --hard origin/<branch>

# Remove large files from history
git filter-branch --tree-filter 'rm -f <large-file>' HEAD
# Or use git-filter-repo (recommended)
```

### Debugging
```bash
# Debug Git commands
GIT_TRACE=1 git <command>
GIT_CURL_VERBOSE=1 git <command>
GIT_SSH_COMMAND="ssh -v" git <command>

# Check file attributes
git check-attr -a <file>

# Verify repository integrity
git fsck
```

## Best Practices

### Commit Messages
```bash
# Good commit message format
<type>(<scope>): <subject>

<body>

<footer>

# Types: feat, fix, docs, style, refactor, test, chore
# Example:
feat(auth): add OAuth2 integration

Implemented Google OAuth2 for user authentication.
Added necessary middleware and updated user model.

Closes #123
```

### Branch Naming
```bash
# Common conventions
feature/description
bugfix/description
hotfix/description
release/version
chore/description

# Examples
feature/user-authentication
bugfix/fix-login-error
hotfix/security-patch
release/v2.0.0
```

### `.gitignore` Patterns
```bash
# Common patterns
*.log
*.tmp
.DS_Store
node_modules/
__pycache__/
.env
.vscode/
.idea/

# Negate patterns
!important.log

# Directory-specific
/build/
src/**/temp/
```

## Git Workflows

### Git Flow
```bash
# Main branches
main/master - Production
develop - Development

# Supporting branches
feature/* - New features
release/* - Release preparation
hotfix/* - Emergency fixes

# Commands
git flow init
git flow feature start <name>
git flow feature finish <name>
```

### GitHub Flow
```bash
# Simple workflow
1. Create branch from main
2. Add commits
3. Open Pull Request
4. Discuss and review
5. Merge to main
6. Deploy
```

### GitLab Flow
```bash
# Environment branches
main - Latest
pre-production - Staging
production - Live

# With release branches
main -> release/x.x -> production
```

## Tools & Extensions

### GUI Clients
- **GitKraken** - Cross-platform Git GUI
- **SourceTree** - Free Git client
- **GitHub Desktop** - Simple GitHub integration
- **Tower** - Powerful Git client
- **Fork** - Fast and friendly

### CLI Tools
- **tig** - Text-mode interface
- **lazygit** - Terminal UI
- **diff-so-fancy** - Better diffs
- **git-extras** - Additional commands
- **hub/gh** - GitHub CLI tools

### VS Code Extensions
- GitLens
- Git Graph
- Git History
- GitHub Pull Requests

### Hooks & Automation
```bash
# Pre-commit hook example
#!/bin/sh
# .git/hooks/pre-commit

# Run tests
npm test

# Check linting
npm run lint

# Prevent commit on failure
if [ $? -ne 0 ]; then
    echo "Tests or linting failed"
    exit 1
fi
```

### Performance Tips
```bash
# Enable filesystem monitor
git config core.fsmonitor true

# Increase buffer size
git config http.postBuffer 524288000

# Enable parallel index
git config index.threads true

# Maintenance
git maintenance start
```

---

*Originally from umitkacar/Git-codes repository*