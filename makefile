pr:
	@which claude > /dev/null || (echo "Error: claude CLI not installed" && exit 1)
	@which gh > /dev/null || (echo "Error: gh CLI not installed" && exit 1)
	@gh auth status > /dev/null 2>&1 || (echo "Error: gh not authenticated. Run 'gh auth login'" && exit 1)
	@echo "Creating a PR, this can take 30s"
	@echo "I will open the Github website for you to review and merge"
	claude -p "Create a PR using the gh-pull-requests skill"
