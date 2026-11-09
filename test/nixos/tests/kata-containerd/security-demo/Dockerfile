FROM busybox:1.36.1

COPY agent-task.sh /opt/agent-task.sh
RUN chmod 0555 /opt/agent-task.sh

USER 1000:1000
ENTRYPOINT ["/opt/agent-task.sh"]
